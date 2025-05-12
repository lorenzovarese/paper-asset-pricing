"""Public interface for the data-aggregation module."""

from .aggregate import DataAggregator, aggregate_from_yaml

__all__ = ["DataAggregator", "aggregate_from_yaml"]
