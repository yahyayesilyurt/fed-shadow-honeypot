from typing import List, Tuple, Dict, Optional, Union

import flwr as fl
from flwr.common import (
    FitRes,
    EvaluateRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate


def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """
    Computes a weighted average of metrics from clients, weighted by sample count.
    Passed to Flower's evaluate_metrics_aggregation_fn and fit_metrics_aggregation_fn
    parameters.

    Args:
        metrics: [(num_examples, {"accuracy": 0.9, "f1_score": 0.85}), ...]
    Returns:
        {"accuracy": <weighted_avg>, "f1_score": <weighted_avg>}
    """
    total_examples = sum(n for n, _ in metrics)
    aggregated = {}

    for key in metrics[0][1].keys():
        aggregated[key] = sum(
            m[key] * n for n, m in metrics if key in m
        ) / total_examples

    return aggregated


class ShadowHoneypotStrategy(fl.server.strategy.FedAvg):
    """
    Custom FL strategy that filters out honeypot client updates.

    How it works:
    - Separates incoming updates each round by the 'is_honeypot' label.
    - Only aggregates updates from real clients using FedAvg.
    - Honeypot updates are silently discarded.
    """

    def __init__(self, num_real_clients: int = 16, **kwargs):
        super().__init__(
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            **kwargs,
        )
        self.num_real_clients = num_real_clients

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Filters honeypot updates and runs FedAvg with real clients.
        """
        real_results     = []
        honeypot_count   = 0

        for client_proxy, fit_res in results:
            if fit_res.metrics.get("is_honeypot", False):
                honeypot_count += 1
            else:
                real_results.append((client_proxy, fit_res))

        # --- Round summary ---
        print(f"\n[Round {server_round}] ── Fit Aggregation ──────────────────────")
        print(f"  Total updates       : {len(results)}")
        print(f"  Honeypot (discarded): {honeypot_count}")
        print(f"  Real (aggregated)   : {len(real_results)}")
        if failures:
            print(f"  Failed clients      : {len(failures)}")

        if not real_results:
            print("  [WARN] No real client updates available, skipping round!")
            return None, {}

        # --- FedAvg (real clients only) ---
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in real_results
        ]
        aggregated_ndarrays  = aggregate(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # --- Collect and log real client metrics ---
        fit_metrics = [
            (fit_res.num_examples, fit_res.metrics)
            for _, fit_res in real_results
            if fit_res.metrics
        ]
        metrics_aggregated = {}
        if fit_metrics:
            # Exclude the is_honeypot boolean from the average calculation
            clean_metrics = [
                (n, {k: v for k, v in m.items() if k != "is_honeypot"})
                for n, m in fit_metrics
            ]
            metrics_aggregated = weighted_average(clean_metrics)
            print(f"  Train Loss (avg)    : {metrics_aggregated.get('train_loss', 'N/A'):.4f}")
            print(f"  Train Acc  (avg)    : {metrics_aggregated.get('train_acc',  'N/A'):.4f}")

        return parameters_aggregated, metrics_aggregated


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Collects and logs evaluate metrics.
        Also filters out honeypots since they send dummy metrics during evaluation.
        """
        real_results = [
            (cp, er) for cp, er in results if er.num_examples > 1
        ]

        if not real_results:
            return None, {}

        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, real_results, failures
        )

        print(f"\n[Round {server_round}] ── Evaluate ────────────────────────────")
        print(f"  Loss (avg)          : {loss_aggregated:.4f}")
        print(f"  Accuracy (avg)      : {metrics_aggregated.get('accuracy',  'N/A'):.4f}")
        print(f"  F1 Score (avg)      : {metrics_aggregated.get('f1_score',  'N/A'):.4f}")

        return loss_aggregated, metrics_aggregated
