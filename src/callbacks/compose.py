from typing import List
import copy


class ComposeCallback:
    """
        Composes multiple callbacks into a single callback. Chaining the return of each callback.
    """

    def __init__(self, callbacks: List):
        self.callbacks = callbacks

    def __call__(self, args) -> None:
        results = copy.deepcopy(args)
        for callback in self.callbacks:
            results = callback(results)

        return results
