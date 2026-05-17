"""Empirical Swap-event replayer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from data.swap_replay import SwapEvent


@dataclass(frozen=True)
class BlockFlow:
    amount0_net: int
    amount1_net: int


class SwapReplayer:
    # cyclic=True wraps the cursor back to start on exhaustion so long PPO
    # rollouts don't hit StopIteration when the historical window is shorter
    # than total timesteps.

    def __init__(
        self,
        events: Sequence[SwapEvent],
        *,
        start_block: int,
        cyclic: bool = False,
    ) -> None:
        if start_block < 0:
            raise ValueError(f"start_block must be >= 0, got {start_block}")
        sorted_events: List[SwapEvent] = sorted(
            events, key=lambda e: (e.block_number, e.log_index)
        )
        self._events: List[SwapEvent] = sorted_events
        self._start_block = int(start_block)
        self._max_event_block = (
            sorted_events[-1].block_number if sorted_events else start_block - 1
        )
        self._cyclic = bool(cyclic)
        self._cursor = 0
        self._current_block = int(start_block)
        self._reset_cursor = self._initial_cursor(sorted_events, start_block)
        self._cursor = self._reset_cursor

    @staticmethod
    def _initial_cursor(events: Sequence[SwapEvent], start_block: int) -> int:
        for i, e in enumerate(events):
            if e.block_number >= start_block:
                return i
        return len(events)

    def reset(self) -> None:
        self._cursor = self._reset_cursor
        self._current_block = self._start_block

    def next_block(self) -> BlockFlow:
        if (
            self._current_block > self._max_event_block + 1
            and self._cursor >= len(self._events)
        ):
            if self._cyclic and self._events:
                self.reset()
            else:
                raise StopIteration(
                    f"SwapReplayer exhausted at block {self._current_block} "
                    f"(max event block {self._max_event_block})"
                )
        net0 = 0
        net1 = 0
        while (
            self._cursor < len(self._events)
            and self._events[self._cursor].block_number == self._current_block
        ):
            ev = self._events[self._cursor]
            net0 += (ev.amount0_in - ev.amount0_out)
            net1 += (ev.amount1_in - ev.amount1_out)
            self._cursor += 1
        self._current_block += 1
        return BlockFlow(amount0_net=int(net0), amount1_net=int(net1))
