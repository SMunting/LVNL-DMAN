"""
This module implements a suite of metrics used to evaluate the quality of
scheduling outcomes in an air‑traffic or airport operations context.

Each function in this module takes as input structured data about flights
and returns a numerical measure.  The metrics follow those described in the
evaluation section of the thesis: predictive accuracy, schedule stability,
throughput utilisation, timeliness of stable schedules and equity/fairness
measures.  All functions operate purely on in‑memory data structures and
do not perform any I/O, so they are suitable for use in tests or as part
of larger analytic pipelines.

The expected data formats are kept simple:

* Times are represented as floating point minutes relative to a common
  epoch (e.g. minutes past midnight) or Python ``datetime`` objects.
* Flight objects are represented as dictionaries with required keys.
* Category labels (for equity) are arbitrary hashable values such as strings.

If your project uses a different data representation (e.g. pandas DataFrames),
these functions can be easily adapted by mapping your data into the expected
structures before calling the metric functions.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Dict, Sequence, Mapping, Any

import numpy as np


def mean_absolute_error(y_pred: Sequence[float], y_true: Sequence[float]) -> float:
    """Return the mean absolute error (MAE) between predicted and true values.

    Parameters
    ----------
    y_pred : Sequence[float]
        Predicted values.
    y_true : Sequence[float]
        Actual values.

    Returns
    -------
    float
        Mean absolute error.
    """
    errors = [abs(p - t) for p, t in zip(y_pred, y_true)]
    return float(sum(errors)) / len(errors) if errors else 0.0


def root_mean_squared_error(y_pred: Sequence[float], y_true: Sequence[float]) -> float:
    """Return the root mean squared error (RMSE) between predicted and true values.

    Parameters
    ----------
    y_pred : Sequence[float]
        Predicted values.
    y_true : Sequence[float]
        Actual values.

    Returns
    -------
    float
        Root mean squared error.
    """
    squared_errors = [(p - t) ** 2 for p, t in zip(y_pred, y_true)]
    return math.sqrt(sum(squared_errors) / len(squared_errors)) if squared_errors else 0.0


def predictive_accuracy(flights: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    """Compute predictive accuracy metrics for a set of flights.

    For each flight we expect the following keys:

    - ``tobt_pred``: predicted target off block time (TOBT)
    - ``aobt``: actual off block time (AOBT)
    - ``scheduled_tot``: scheduled take‑off time
    - ``atot``: actual take‑off time (ATOT)

    The function returns a dictionary with mean absolute error and root mean
    squared error for both the TOBT prediction and the scheduled take‑off.

    Parameters
    ----------
    flights : Iterable[Mapping[str, Any]]
        Iterable of flight dictionaries with prediction and actual timing information.

    Returns
    -------
    Dict[str, float]
        A mapping containing 'mae_tobt', 'rmse_tobt', 'mae_tot', and 'rmse_tot'.
    """
    tobt_pred = []
    aobt = []
    sched_tot = []
    atot = []
    for flight in flights:
        # Skip flights missing data
        try:
            tobt_pred.append(float(flight["tobt_pred"]))
            aobt.append(float(flight["aobt"]))
            sched_tot.append(float(flight["scheduled_tot"]))
            atot.append(float(flight["atot"]))
        except (KeyError, ValueError):
            continue
    return {
        "mae_tobt": mean_absolute_error(tobt_pred, aobt),
        "rmse_tobt": root_mean_squared_error(tobt_pred, aobt),
        "mae_tot": mean_absolute_error(sched_tot, atot),
        "rmse_tot": root_mean_squared_error(sched_tot, atot),
    }


def schedule_stability(flights: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    """Compute schedule stability metrics for each flight.

    Each flight dictionary must contain a list of slot assignment changes keyed
    by ``assignments``.  Each entry in the list should be a tuple or list of
    the form (timestamp, slot_time).  The sequence must be ordered by the
    time the change occurred.  The first element of the sequence represents
    the initial assignment (at planning time) and subsequent entries represent
    changes.

    The stability metrics computed are:

    - ``avg_change_minutes``: average absolute change in slot (minutes) per
      change event across all flights.
    - ``avg_resequences``: average number of resequencing shifts experienced
      across all flights.

    Parameters
    ----------
    flights : Iterable[Mapping[str, Any]]
        Iterable of flight dictionaries with assignment history.

    Returns
    -------
    Dict[str, float]
        Mapping containing 'avg_change_minutes' and 'avg_resequences'.
    """
    total_change_mins = 0.0
    total_changes = 0
    total_resequences = 0
    num_flights = 0
    for flight in flights:
        assignments = flight.get("assignments", [])
        # ignore flights with fewer than two assignments
        if not assignments or len(assignments) < 2:
            continue
        num_flights += 1
        prev_slot = assignments[0][1]
        for _, current_slot in assignments[1:]:
            # compute absolute change in minutes between previous and current slot
            change_minutes = abs(float(current_slot) - float(prev_slot))
            total_change_mins += change_minutes
            total_changes += 1
            prev_slot = current_slot
        # number of resequencing shifts is changes minus one initial assignment
        total_resequences += (len(assignments) - 1)
    avg_change = total_change_mins / total_changes if total_changes else 0.0
    avg_reseq = total_resequences / num_flights if num_flights else 0.0
    return {
        "avg_change_minutes": avg_change,
        "avg_resequences": avg_reseq,
    }


def throughput_utilisation(used_slots: int, runway_capacity: int) -> float:
    """Compute the throughput utilisation of the runway.

    Throughput utilisation is defined as the ratio of used runway slots to the
    declared runway capacity for a given period (e.g. an hour).  The result is
    expressed as a percentage between 0 and 100.  If the capacity is zero,
    utilisation is reported as 0.

    Parameters
    ----------
    used_slots : int
        Number of runway slots utilised during the period.
    runway_capacity : int
        Declared runway capacity (maximum possible slots) for the same period.

    Returns
    -------
    float
        Utilisation percentage (0–100).
    """
    if runway_capacity <= 0:
        return 0.0
    utilisation = (used_slots / runway_capacity) * 100.0
    return float(utilisation)


def timeliness_of_stable_schedule(
    flights: Iterable[Mapping[str, Any]],
    thresholds: Sequence[int],
) -> Dict[int, float]:
    """Compute the timeliness of achieving a stable slot assignment.

    For each lead time threshold ``tau`` (in minutes), the metric measures
    the proportion of flights whose assigned slot does not change after
    ``aobt - tau``.  In other words, at time ``aobt - tau``, the slot
    assignment remains identical to the final assignment at actual off block
    time.

    Each flight dictionary must provide ``aobt`` and ``assignments`` as in
    ``schedule_stability``.

    Parameters
    ----------
    flights : Iterable[Mapping[str, Any]]
        Iterable of flight dictionaries with assignment history and AOBT.
    thresholds : Sequence[int]
        Lead time thresholds in minutes.  Should be sorted in descending order
        if you wish to process longer lead times first, but not required.

    Returns
    -------
    Dict[int, float]
        Mapping from each threshold to the proportion of flights (0–1) whose
        slot remains stable after that lead time.
    """
    # normalise to list for multiple passes
    flights_list = list(flights)
    results = {}
    for tau in thresholds:
        stable_count = 0
        total_count = 0
        for flight in flights_list:
            aobt = flight.get("aobt")
            assignments = flight.get("assignments")
            if aobt is None or not assignments:
                continue
            # time after which we require stability
            cutoff_time = float(aobt) - float(tau)
            # find the last assignment strictly before cutoff_time
            # assignments must be sorted by change timestamp
            last_slot = assignments[-1][1]
            slot_at_cutoff = last_slot
            for ts, slot_time in assignments:
                if float(ts) <= cutoff_time:
                    slot_at_cutoff = slot_time
                else:
                    break
            total_count += 1
            if slot_at_cutoff == last_slot:
                stable_count += 1
        proportion = (stable_count / total_count) if total_count else 0.0
        results[int(tau)] = proportion
    return results


def gini_coefficient(values: Sequence[float]) -> float:
    """Compute the Gini coefficient of a list of numerical values.

    The Gini coefficient measures inequality in a distribution.  A value of
    zero indicates perfect equality, while a value of one indicates maximal
    inequality.

    Parameters
    ----------
    values : Sequence[float]
        List or array of values.  Negative values are allowed but will
        increase the inequality measure.

    Returns
    -------
    float
        The Gini coefficient, between 0 and 1 (inclusive).
    """
    array = np.array(values, dtype=float)
    if array.size == 0:
        return 0.0
    # ensure non-negative by shifting if necessary
    array -= array.min()
    total = array.sum()
    if total == 0:
        return 0.0
    # sort and compute Lorenz curve
    sorted_vals = np.sort(array)
    cumulative = np.cumsum(sorted_vals)
    # Gini computation: area between line of equality and Lorenz curve
    n = sorted_vals.size
    # relative mean absolute difference times 0.5
    gini = (n + 1 - 2 * (cumulative / total).sum()) / n
    return float(gini)


def between_group_variance(values: Sequence[float], labels: Sequence[Any]) -> float:
    """Compute the variance of mean values across groups.

    This metric measures how much the average value differs between
    categories.  It is one component of equity analysis: a high between‑group
    variance indicates that some groups experience systematically larger
    delays (or other metric) than others.

    Parameters
    ----------
    values : Sequence[float]
        Numeric values (e.g. slot deviations) associated with each flight.
    labels : Sequence[Any]
        Group labels (e.g. airline, destination, aircraft class) for the
        corresponding values.

    Returns
    -------
    float
        The sample variance of the group means.  If no variance can be
        computed (e.g. fewer than two groups), zero is returned.
    """
    group_sums: Dict[Any, float] = defaultdict(float)
    group_counts: Dict[Any, int] = defaultdict(int)
    for v, lbl in zip(values, labels):
        group_sums[lbl] += float(v)
        group_counts[lbl] += 1
    group_means = [group_sums[lbl] / group_counts[lbl] for lbl in group_counts]
    if len(group_means) < 2:
        return 0.0
    return float(np.var(group_means, ddof=1))  # sample variance


def equity_measures(
    flights: Iterable[Mapping[str, Any]], category_key: str
) -> Dict[str, float]:
    """Compute equity metrics across flights for a given category.

    The function calculates both the Gini coefficient of slot deviations and
    the between‑group variance of the mean slot deviations.  A slot deviation
    is defined here as the difference between the scheduled take‑off time and
    the actual take‑off time (positive values indicate delays).

    Each flight dictionary must contain:
    - ``scheduled_tot``: scheduled take‑off time
    - ``atot``: actual take‑off time
    - category_key: the category label (e.g. airline, destination, class)

    Parameters
    ----------
    flights : Iterable[Mapping[str, Any]]
        Iterable of flight dictionaries with schedule and category information.
    category_key : str
        The key in each flight dict used to group flights (e.g. 'airline').

    Returns
    -------
    Dict[str, float]
        A mapping with 'gini' and 'between_group_var'.
    """
    deviations = []
    labels = []
    for flight in flights:
        try:
            sched = float(flight["scheduled_tot"])
            actual = float(flight["atot"])
            deviations.append(actual - sched)
            labels.append(flight[category_key])
        except (KeyError, ValueError):
            continue
    return {
        "gini": gini_coefficient(deviations),
        "between_group_var": between_group_variance(deviations, labels),
    }
