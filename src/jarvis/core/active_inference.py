"""
JARVIS-PRIME Active Inference Engine
=======================================

Implements the Free Energy Principle (Karl Friston) for:
1. Perception: Update beliefs about hidden states (state estimation)
2. Action: Select actions that minimize expected free energy
3. Learning: Update generative model parameters

Core equation: F = E_q[log q(s) - log p(o,s)]
             = Complexity - Accuracy
             = KL[q(s) || p(s)] - E_q[log p(o|s)]

Phase 3: Discrete state space, simplified free energy
Phase 4+: Continuous variational inference
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


class BeliefState:
    """
    Represents an agent's beliefs about the world as a probability distribution.
    Uses a categorical distribution over discrete states.
    """

    def __init__(self, n_states: int = 8, prior: np.ndarray | None = None):
        self.n_states = n_states
        if prior is not None:
            self.distribution = prior / prior.sum()
        else:
            self.distribution = np.ones(n_states) / n_states  # Uniform prior

    def entropy(self) -> float:
        """Shannon entropy of the belief distribution."""
        p = self.distribution[self.distribution > 1e-15]
        return float(-np.sum(p * np.log(p)))

    def max_entropy(self) -> float:
        """Maximum possible entropy (uniform distribution)."""
        return float(np.log(self.n_states))

    def certainty(self) -> float:
        """How certain the agent is (1 - normalized entropy)."""
        return 1.0 - self.entropy() / max(self.max_entropy(), 1e-10)

    def most_likely_state(self) -> int:
        """Return the most probable state."""
        return int(np.argmax(self.distribution))

    def update(self, likelihood: np.ndarray) -> None:
        """Bayesian update: posterior = prior * likelihood / Z."""
        posterior = self.distribution * likelihood
        total = posterior.sum()
        if total > 0:
            self.distribution = posterior / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_states": self.n_states,
            "distribution": [round(float(p), 4) for p in self.distribution],
            "entropy": round(self.entropy(), 4),
            "certainty": round(self.certainty(), 4),
            "most_likely_state": self.most_likely_state(),
        }


class GenerativeModel:
    """
    Generative model: p(observations, states) = p(obs|state) * p(state).

    Components:
    - A (likelihood): p(observation | hidden state)
    - B (transition): p(state_t+1 | state_t, action)
    - C (preferences): preferred observations (goals)
    - D (prior): p(state_0) initial state prior
    """

    def __init__(self, n_states: int = 8, n_obs: int = 8, n_actions: int = 4):
        self.n_states = n_states
        self.n_obs = n_obs
        self.n_actions = n_actions

        # Likelihood: p(obs | state) — each state has a different emission pattern
        self.A = np.eye(n_obs, n_states) * 0.7 + 0.3 / n_obs
        self.A /= self.A.sum(axis=0, keepdims=True)

        # Transition: p(state' | state, action) — one matrix per action
        self.B = []
        for a in range(n_actions):
            T = np.eye(n_states) * 0.6
            # Each action shifts probability to different states
            for i in range(n_states):
                T[(i + a + 1) % n_states, i] += 0.4
            T /= T.sum(axis=0, keepdims=True)
            self.B.append(T)

        # Preferences: which observations are desirable
        self.C = np.zeros(n_obs)
        self.C[0] = 2.0   # Prefer state 0 (goal state)
        self.C[1] = 1.0   # Also like state 1

        # Prior: initial state distribution
        self.D = np.ones(n_states) / n_states


class ActiveInferenceAgent:
    """
    Active Inference agent that:
    1. Perceives: updates beliefs about states given observations
    2. Plans: evaluates actions by expected free energy
    3. Acts: selects actions that minimize expected free energy
    4. Learns: updates model parameters over time
    """

    def __init__(self, n_states: int = 8, n_obs: int = 8, n_actions: int = 4):
        self.model = GenerativeModel(n_states, n_obs, n_actions)
        self.beliefs = BeliefState(n_states, self.model.D.copy())
        self.action_history: list[int] = []
        self.observation_history: list[int] = []
        self.free_energy_history: list[float] = []

    def perceive(self, observation: int) -> dict[str, Any]:
        """
        Update beliefs given a new observation using Bayesian inference.

        posterior(s) ~ likelihood(o|s) * prior(s)
        """
        self.observation_history.append(observation)

        # Likelihood of observation under each state
        likelihood = self.model.A[observation, :]

        # Bayesian update
        prior = self.beliefs.to_dict()
        self.beliefs.update(likelihood)
        posterior = self.beliefs.to_dict()

        # Compute variational free energy
        fe = self._variational_free_energy(observation)
        self.free_energy_history.append(fe)

        return {
            "observation": observation,
            "prior_certainty": prior["certainty"],
            "posterior_certainty": posterior["certainty"],
            "most_likely_state": posterior["most_likely_state"],
            "free_energy": round(fe, 4),
            "belief_update": posterior,
        }

    def plan(self, planning_horizon: int = 3) -> dict[str, Any]:
        """
        Evaluate all possible actions using expected free energy.

        G(a) = E_q[log q(s') - log p(o', s')]
             = ambiguity + risk

        Lower G is better.
        """
        n_actions = self.model.n_actions
        action_values = []

        for action in range(n_actions):
            G = self._expected_free_energy(action, horizon=planning_horizon)
            action_values.append({
                "action": action,
                "expected_free_energy": round(G, 4),
            })

        # Sort by EFE (lower is better)
        action_values.sort(key=lambda x: x["expected_free_energy"])

        # Action probabilities via softmax
        G_values = np.array([av["expected_free_energy"] for av in action_values])
        G_shifted = G_values - G_values.min()
        probs = np.exp(-G_shifted)
        probs /= probs.sum()

        for i, av in enumerate(action_values):
            av["probability"] = round(float(probs[i]), 4)

        return {
            "planning_horizon": planning_horizon,
            "action_evaluations": action_values,
            "best_action": action_values[0]["action"],
            "best_EFE": action_values[0]["expected_free_energy"],
        }

    def act(self, action: int | None = None) -> dict[str, Any]:
        """
        Select and execute an action.
        If no action provided, selects optimal via planning.
        """
        if action is None:
            plan = self.plan()
            action = plan["best_action"]

        self.action_history.append(action)

        # Update beliefs through predicted transition
        transition = self.model.B[action]
        predicted = transition @ self.beliefs.distribution
        self.beliefs.distribution = predicted

        return {
            "action_taken": action,
            "predicted_state": self.beliefs.most_likely_state(),
            "certainty": round(self.beliefs.certainty(), 4),
        }

    def _variational_free_energy(self, observation: int) -> float:
        """
        Compute variational free energy:
        F = -E_q[log p(o|s)] + KL[q(s) || p(s)]
          = -accuracy + complexity
        """
        q = self.beliefs.distribution
        prior = self.model.D

        # Accuracy: expected log-likelihood
        log_likelihood = np.log(self.model.A[observation, :] + 1e-10)
        accuracy = float(np.dot(q, log_likelihood))

        # Complexity: KL divergence
        kl = float(np.sum(q * np.log((q + 1e-10) / (prior + 1e-10))))

        return -accuracy + kl

    def _expected_free_energy(self, action: int, horizon: int = 1) -> float:
        """
        Compute expected free energy for an action.

        G = ambiguity + risk
        Ambiguity: H[p(o|s)] — uncertainty about observations
        Risk: KL[q(o) || p_pref(o)] — divergence from preferences
        """
        transition = self.model.B[action]
        predicted_state = transition @ self.beliefs.distribution

        # Expected observations
        predicted_obs = self.model.A @ predicted_state

        # Ambiguity: conditional entropy of observations given states
        ambiguity = 0.0
        for s in range(self.model.n_states):
            if predicted_state[s] > 1e-10:
                obs_given_s = self.model.A[:, s]
                H_obs = -np.sum(obs_given_s * np.log(obs_given_s + 1e-10))
                ambiguity += predicted_state[s] * H_obs

        # Risk: divergence from preferred observations
        log_pref = self.model.C  # log-preferences
        risk = -float(np.dot(predicted_obs, log_pref))

        return ambiguity + risk

    def run_episode(self, n_steps: int = 20) -> dict[str, Any]:
        """
        Run a full active inference episode.
        Environment is simulated from the generative model.
        """
        # Start from random state
        true_state = np.random.choice(self.model.n_states)
        self.beliefs = BeliefState(self.model.n_states, self.model.D.copy())

        episode_log = []

        for step in range(n_steps):
            # Generate observation from true state
            obs_probs = self.model.A[:, true_state]
            observation = np.random.choice(self.model.n_obs, p=obs_probs)

            # Perceive
            perception = self.perceive(observation)

            # Plan and act
            plan = self.plan(planning_horizon=2)
            action_result = self.act(plan["best_action"])

            # Environment transition
            trans_probs = self.model.B[plan["best_action"]][:, true_state]
            true_state = np.random.choice(self.model.n_states, p=trans_probs)

            episode_log.append({
                "step": step,
                "true_state": int(true_state),
                "believed_state": perception["most_likely_state"],
                "action": plan["best_action"],
                "free_energy": perception["free_energy"],
            })

        # Accuracy of state estimation
        correct = sum(1 for e in episode_log if e["true_state"] == e["believed_state"])

        return {
            "n_steps": n_steps,
            "state_estimation_accuracy": round(correct / n_steps, 4),
            "final_free_energy": episode_log[-1]["free_energy"],
            "mean_free_energy": round(
                float(np.mean([e["free_energy"] for e in episode_log])), 4
            ),
            "fe_trend": "DECREASING" if episode_log[-1]["free_energy"] < episode_log[0]["free_energy"] else "STABLE",
            "log_sample": episode_log[:5],
        }

    def stats(self) -> dict[str, Any]:
        return {
            "model": {
                "n_states": self.model.n_states,
                "n_observations": self.model.n_obs,
                "n_actions": self.model.n_actions,
            },
            "beliefs": self.beliefs.to_dict(),
            "history": {
                "observations": len(self.observation_history),
                "actions": len(self.action_history),
                "free_energy_samples": len(self.free_energy_history),
            },
            "mean_free_energy": round(
                float(np.mean(self.free_energy_history)) if self.free_energy_history else 0, 4
            ),
        }
