"""A Priority Queue implementation.

Copyright (C) 2023 Michael Kuchnik. All Right Reserved.
Licensed under the Apache License, Version 2.0

Copyright 2014 Red Blob Games <redblobgames@gmail.com>
License: Apache v2.0 <http://www.apache.org/licenses/LICENSE-2.0.html>
"""
import heapq
import typing


class PriorityQueue:
    """A priority queue wrapper around heapq.

    Modified from:
    https://www.redblobgames.com/pathfinding/a-star/implementation.html
    """

    def __init__(self, data=None):
        """Initialize the PriorityQueue."""
        if data is None:
            elements = []
        else:
            elements = list(data)
            heapq.heapify(elements)
        self.elements: typing.List[typing.Tuple[float, typing.T]] = elements

    def empty(self) -> bool:
        """Check if empty."""
        return not self.elements

    def put(self, priority_item: typing.Tuple[float, typing.T]):
        """Put an element into the priority queue."""
        priority, item = priority_item
        heapq.heappush(self.elements, (priority, item))

    def get(self) -> typing.Tuple[float, typing.T]:
        """Pop an element off the priority queue."""
        return heapq.heappop(self.elements)

    def peak(self) -> typing.Tuple[float, typing.T]:
        """Peak the element at the head of the priority queue."""
        return self.elements[0]

    def __str__(self) -> str:
        """Return a string representation of the queue."""
        return "{{PQ:{}}}".format(self.elements)

    def __repr__(self) -> str:
        """Return a string representation of the queue."""
        return "{{PQ:{}}}".format(self.elements)

    def __len__(self) -> int:
        """Return the length of the queue."""
        return len(self.elements)
