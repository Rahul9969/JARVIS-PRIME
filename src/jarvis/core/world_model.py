"""
JARVIS-PRIME World Model (JEPA-Inspired Predictive Coding)
=============================================================

Implements a predictive coding framework inspired by JEPA
(Joint Embedding Predictive Architecture):

1. State Encoder: Compress observations into latent states
2. Predictor: Predict future latent states from current state + action
3. World Simulator: Simulate outcomes of hypothetical actions
4. Anomaly Detector: Identify unexpected observations

Phase 3: NumPy-based latent state tracking
Phase 4+: Neural predictive model with learned embeddings
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np


class LatentState:
    """
    Represents a compressed world state in latent space.
    Think of it as JARVIS's internal model of "how the world is right now."
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.vector = np.zeros(dim)
        self.timestamp = time.time()
        self.confidence = 1.0
        self.source = "initial"

    def update(self, observation: np.ndarray, learning_rate: float = 0.1) -> None:
        """Update latent state with new observation via EMA."""
        if len(observation) != self.dim:
            # Project to correct dimension
            observation = np.resize(observation, self.dim)

        self.vector = (1 - learning_rate) * self.vector + learning_rate * observation
        self.timestamp = time.time()

    def similarity(self, other: "LatentState") -> float:
        """Cosine similarity between two latent states."""
        norm_a = np.linalg.norm(self.vector)
        norm_b = np.linalg.norm(other.vector)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(self.vector, other.vector) / (norm_a * norm_b))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dim,
            "norm": round(float(np.linalg.norm(self.vector)), 4),
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "nonzero_dims": int(np.count_nonzero(self.vector > 0.01)),
        }


class WorldModel:
    """
    JEPA-inspired world model for JARVIS-PRIME.

    Maintains an internal model of the system's state and can:
    1. Encode observations into latent space
    2. Predict future states
    3. Detect anomalies (prediction errors)
    4. Simulate hypothetical scenarios
    """

    def __init__(self, state_dim: int = 64, history_size: int = 100):
        self.state_dim = state_dim
        self.current_state = LatentState(dim=state_dim)
        self.history: list[dict[str, Any]] = []
        self.history_size = history_size

        # Prediction model: linear predictor (Phase 3)
        # Phase 4+: Replace with neural network
        self.transition_matrix = np.eye(state_dim) * 0.95  # Slight decay
        self.prediction_errors: list[float] = []

        # Domain state tracking
        self.domain_states: dict[str, LatentState] = {}

    def encode_observation(
        self,
        observation: dict[str, Any],
        domain: str = "general",
    ) -> LatentState:
        """
        Encode a structured observation into latent space.
        Uses feature hashing for dictionary -> vector mapping.
        """
        vec = np.zeros(self.state_dim)

        # Hash dictionary keys and values into the latent vector
        for key, value in observation.items():
            key_hash = hash(key) % self.state_dim
            if isinstance(value, (int, float)):
                vec[key_hash] += float(value)
            elif isinstance(value, str):
                val_hash = hash(value) % self.state_dim
                vec[val_hash] += 1.0
            elif isinstance(value, bool):
                vec[key_hash] += 1.0 if value else -1.0
            elif isinstance(value, list):
                vec[key_hash] += len(value)

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        # Update current state
        self.current_state.update(vec)
        self.current_state.source = domain

        # Update domain-specific state
        if domain not in self.domain_states:
            self.domain_states[domain] = LatentState(dim=self.state_dim)
        self.domain_states[domain].update(vec)

        # Record in history
        self.history.append({
            "timestamp": time.time(),
            "domain": domain,
            "state_norm": float(np.linalg.norm(self.current_state.vector)),
        })
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]

        return self.current_state

    def predict_next_state(self, action: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Predict the next world state given an action.
        Uses linear transition model (Phase 3).
        """
        # Predict next state
        predicted_vec = self.transition_matrix @ self.current_state.vector

        if action:
            # Encode action and add perturbation
            action_vec = np.zeros(self.state_dim)
            for key, value in action.items():
                idx = hash(key) % self.state_dim
                if isinstance(value, (int, float)):
                    action_vec[idx] += float(value)
            norm = np.linalg.norm(action_vec)
            if norm > 0:
                action_vec /= norm
            predicted_vec += 0.1 * action_vec

        predicted_state = LatentState(dim=self.state_dim)
        predicted_state.vector = predicted_vec
        predicted_state.source = "prediction"

        return {
            "predicted_state": predicted_state.to_dict(),
            "similarity_to_current": round(
                self.current_state.similarity(predicted_state), 4
            ),
            "action_applied": action is not None,
            "model": "linear_transition",
        }

    def detect_anomaly(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Detect if an observation is anomalous (high prediction error).
        """
        # Get predicted state
        prediction = self.predict_next_state()

        # Encode observation
        obs_state = LatentState(dim=self.state_dim)
        vec = np.zeros(self.state_dim)
        for key, value in observation.items():
            idx = hash(key) % self.state_dim
            if isinstance(value, (int, float)):
                vec[idx] += float(value)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        obs_state.vector = vec

        # Prediction error
        error = float(np.linalg.norm(
            obs_state.vector - self.transition_matrix @ self.current_state.vector
        ))

        self.prediction_errors.append(error)
        if len(self.prediction_errors) > 100:
            self.prediction_errors = self.prediction_errors[-100:]

        # Anomaly threshold: mean + 2*std
        mean_error = np.mean(self.prediction_errors)
        std_error = np.std(self.prediction_errors) if len(self.prediction_errors) > 1 else 1.0
        threshold = mean_error + 2 * std_error

        is_anomaly = error > threshold and len(self.prediction_errors) > 5

        return {
            "prediction_error": round(error, 4),
            "threshold": round(float(threshold), 4),
            "mean_error": round(float(mean_error), 4),
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": round(float((error - mean_error) / max(std_error, 1e-10)), 2),
        }

    def simulate_scenario(
        self,
        actions: list[dict[str, Any]],
        steps: int | None = None,
    ) -> dict[str, Any]:
        """
        Simulate a sequence of hypothetical actions.
        Returns predicted trajectory without modifying actual state.
        """
        steps = steps or len(actions)
        sim_vec = self.current_state.vector.copy()
        trajectory = []

        for i in range(min(steps, len(actions))):
            # Apply transition
            sim_vec = self.transition_matrix @ sim_vec

            # Apply action perturbation
            action_vec = np.zeros(self.state_dim)
            for key, value in actions[i].items():
                idx = hash(key) % self.state_dim
                if isinstance(value, (int, float)):
                    action_vec[idx] += float(value)
            norm = np.linalg.norm(action_vec)
            if norm > 0:
                sim_vec += 0.1 * action_vec / norm

            trajectory.append({
                "step": i + 1,
                "state_norm": round(float(np.linalg.norm(sim_vec)), 4),
                "divergence": round(float(np.linalg.norm(
                    sim_vec - self.current_state.vector
                )), 4),
            })

        return {
            "scenario_steps": len(trajectory),
            "trajectory": trajectory,
            "final_divergence": trajectory[-1]["divergence"] if trajectory else 0,
            "model": "linear_simulation",
        }

    def stats(self) -> dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "current_state": self.current_state.to_dict(),
            "history_length": len(self.history),
            "domains_tracked": list(self.domain_states.keys()),
            "prediction_errors_recorded": len(self.prediction_errors),
            "mean_prediction_error": round(
                float(np.mean(self.prediction_errors)) if self.prediction_errors else 0, 4
            ),
            "model_type": "linear_transition (Phase 3)",
        }
