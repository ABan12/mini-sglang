from __future__ import annotations

import multiprocessing as mp
from itertools import groupby
from typing import List

import torch
from minisgl.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeAbortMsg,
    DetokenizeMsg,
    RejectMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger, load_tokenizer


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    model_source: str = "huggingface",
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    tokenizer = load_tokenizer(tokenizer_path)
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    tokenize_manager = TokenizeManager(tokenizer)
    detokenize_manager = DetokenizeManager(tokenizer)

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            pending_msg = _unwrap_msg(recv_listener.get())
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            # Batch adjacent messages without moving cancellations past later work.
            for msg_type, group in groupby(pending_msg, key=type):
                msgs = list(group)
                if msg_type is DetokenizeMsg:
                    replies = detokenize_manager.detokenize(msgs)
                    batch_output = BatchFrontendMsg(
                        data=[
                            UserReply(
                                uid=msg.uid,
                                incremental_output=reply,
                                finished=msg.finished,
                                finish_reason=msg.finish_reason,
                                prompt_tokens=msg.prompt_tokens,
                                completion_tokens=msg.completion_tokens,
                            )
                            for msg, reply in zip(msgs, replies, strict=True)
                        ]
                    )
                    if len(batch_output.data) == 1:
                        batch_output = batch_output.data[0]
                    send_frontend.put(batch_output)
                elif msg_type is TokenizeMsg:
                    tensors = tokenize_manager.tokenize(msgs)
                    batch_output = BatchBackendMsg(
                        data=[
                            UserMsg(uid=msg.uid, input_ids=t, sampling_params=msg.sampling_params)
                            for msg, t in zip(msgs, tensors, strict=True)
                        ]
                    )
                    if len(batch_output.data) == 1:
                        batch_output = batch_output.data[0]
                    send_backend.put(batch_output)
                elif msg_type is AbortMsg:
                    for msg in msgs:
                        detokenize_manager.abort(msg.uid)
                    batch_output = BatchBackendMsg(
                        data=[AbortBackendMsg(uid=msg.uid) for msg in msgs]
                    )
                    if len(batch_output.data) == 1:
                        batch_output = batch_output.data[0]
                    send_backend.put(batch_output)
                elif msg_type is DetokenizeAbortMsg:
                    for msg in msgs:
                        detokenize_manager.abort(msg.uid)
                elif msg_type is RejectMsg:
                    batch_output = BatchFrontendMsg(
                        data=[
                            UserReply(uid=msg.uid, incremental_output="", finished=True, error=msg.error)
                            for msg in msgs
                        ]
                    )
                    if len(batch_output.data) == 1:
                        batch_output = batch_output.data[0]
                    send_frontend.put(batch_output)
                else:
                    raise TypeError(f"Unknown tokenizer message: {msg_type}")
    except KeyboardInterrupt:
        pass
