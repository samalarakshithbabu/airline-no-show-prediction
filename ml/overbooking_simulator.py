"""
Overbooking Simulator - Monte Carlo simulation for airline overbooking optimization
"""

import os
import sys
import numpy as np
from dataclasses import dataclass
import math
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class CostParameters:
    """Cost parameters for overbooking simulation."""
    compensation_cost: float = 1500.0       # Cost to compensate denied boarding
    empty_seat_cost: float = 300.0          # Cost per empty seat
    denied_boarding_cost: float = 5000.0    # Applied per denied passenger
    
    @classmethod
    def from_dict(cls, data: Dict = None) -> 'CostParameters':
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError('cost parameters must be an object')
        defaults = cls()
        values = {}
        for field in ('compensation_cost', 'empty_seat_cost', 'denied_boarding_cost'):
            value = data.get(field, getattr(defaults, field))
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{field} must be a finite non-negative number')
            values[field] = float(value)
        return cls(**values)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'compensation_cost': self.compensation_cost,
            'empty_seat_cost': self.empty_seat_cost,
            'denied_boarding_cost': self.denied_boarding_cost
        }


@dataclass
class OverbookingScenario:
    """Results of an overbooking scenario simulation."""
    overbooking_level: int
    capacity: int
    bookings: int
    expected_shows: float
    expected_noshows: float
    expected_denied: float
    expected_empty_seats: float
    expected_revenue: float
    expected_cost: float
    expected_profit: float
    probability_denied: float
    profit_std: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overbooking_level': self.overbooking_level,
            'capacity': self.capacity,
            'bookings': self.bookings,
            'expected_shows': round(self.expected_shows, 2),
            'expected_noshows': round(self.expected_noshows, 2),
            'expected_denied': round(self.expected_denied, 2),
            'expected_empty_seats': round(self.expected_empty_seats, 2),
            'expected_revenue': round(self.expected_revenue, 2),
            'expected_cost': round(self.expected_cost, 2),
            'expected_profit': round(self.expected_profit, 2),
            'probability_denied': round(self.probability_denied, 4),
            'profit_std': round(self.profit_std, 2)
        }


class OverbookingSimulator:
    """
    Monte Carlo simulator for airline overbooking decisions.
    
    Uses binomial distribution to model passenger show-up behavior
    and calculate expected outcomes for different overbooking levels.
    """
    
    def __init__(self, cost_params: CostParameters = None, random_seed: int = None):
        """
        Initialize overbooking simulator.
        
        Args:
            cost_params: Cost parameters for the simulation
            random_seed: Random seed for reproducibility
        """
        self.cost_params = cost_params or CostParameters()
        self.rng = np.random.default_rng(random_seed)
    
    def calculate_show_probability(self, noshow_prob: float, passenger_features: Dict = None) -> float:
        """
        Calculate passenger show-up probability.
        
        For now, uses the provided no-show probability.
        In production, this could use ML predictions.
        
        Args:
            noshow_prob: Base no-show probability
            passenger_features: Optional passenger features for ML-based adjustment
            
        Returns:
            Show-up probability (1 - no-show probability)
        """
        if passenger_features is None:
            return 1.0 - noshow_prob
        
        # Adjust based on passenger features
        # This is a simplified version - in production, use ML model
        show_prob = 1.0 - noshow_prob
        
        # Adjust for loyalty tier
        loyalty_bonus = passenger_features.get('loyalty_tier_encoded', 0) * 0.02
        show_prob += loyalty_bonus
        
        # Adjust for booking lead time
        lead_time = passenger_features.get('booking_lead_time', 14)
        if lead_time > 30:
            show_prob += 0.03  # Longer advance booking = more likely to show
        elif lead_time < 3:
            show_prob -= 0.05  # Short notice = less likely to show
        
        # Clamp to reasonable range
        return min(0.99, max(0.5, show_prob))
    
    def simulate_shows(self, bookings: int, show_prob: float, n_simulations: int = 10000) -> np.ndarray:
        """
        Simulate passenger show-ups using binomial distribution.
        
        Args:
            bookings: Number of bookings
            show_prob: Probability each passenger shows up
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Array of number of passengers showing up per simulation
        """
        return self.rng.binomial(bookings, show_prob, n_simulations)
    
    def calculate_expected_values(self,
                                   bookings: int,
                                   show_prob: float,
                                   capacity: int,
                                   ticket_price: float,
                                   n_simulations: int = 10000) -> Dict[str, float]:
        """
        Calculate expected values for a given booking situation.
        
        Args:
            bookings: Number of bookings
            show_prob: Probability of showing up
            capacity: Aircraft capacity
            ticket_price: Average ticket price
            n_simulations: Number of simulations
            
        Returns:
            Dictionary of expected values
        """
        # Simulate shows
        shows = self.simulate_shows(bookings, show_prob, n_simulations)
        
        # Calculate metrics
        expected_shows = shows.mean()
        expected_noshows = bookings - expected_shows
        
        # Denied boarding: passengers showing up beyond capacity
        denied = np.maximum(0, shows - capacity)
        expected_denied = denied.mean()
        prob_denied = (denied > 0).mean()
        
        # Empty seats: capacity minus shows (if shows < capacity)
        empty = np.maximum(0, capacity - shows)
        expected_empty_seats = empty.mean()
        
        # Financial calculations
        # Ticket revenue is assumed collected for every confirmed booking, including no-shows.
        expected_revenue = bookings * ticket_price
        
        # Costs
        compensation = denied * self.cost_params.compensation_cost
        expected_compensation = compensation.mean()
        
        empty_seat_cost = empty * self.cost_params.empty_seat_cost
        expected_empty_cost = empty_seat_cost.mean()
        
        denied_boarding_incidents = denied * self.cost_params.denied_boarding_cost
        total_cost = compensation + denied_boarding_incidents + empty_seat_cost
        expected_cost = total_cost.mean()
        
        # Profit (revenue - costs)
        profit = expected_revenue - total_cost
        expected_profit = profit.mean()
        profit_std = profit.std()
        
        return {
            'expected_shows': expected_shows,
            'expected_noshows': expected_noshows,
            'expected_denied': expected_denied,
            'expected_empty_seats': expected_empty_seats,
            'expected_revenue': expected_revenue,
            'expected_cost': expected_cost,
            'expected_profit': expected_profit,
            'profit_std': profit_std,
            'probability_denied': prob_denied
        }
    
    def simulate_scenario(self,
                          capacity: int,
                          bookings: int,
                          avg_noshow_prob: float,
                          ticket_price: float,
                          max_overbooking: int = 15,
                          n_simulations: int = 10000) -> List[OverbookingScenario]:
        """
        Run simulation for multiple overbooking levels.
        
        Args:
            capacity: Aircraft capacity
            bookings: Current number of bookings
            avg_noshow_prob: Average no-show probability for the flight
            ticket_price: Average ticket price
            max_overbooking: Maximum overbooking level to simulate
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            List of scenarios for each overbooking level
        """
        scenarios = []
        show_prob = 1.0 - avg_noshow_prob
        
        for overbooking in range(0, max_overbooking + 1):
            total_bookings = bookings + overbooking
            
            metrics = self.calculate_expected_values(
                bookings=total_bookings,
                show_prob=show_prob,
                capacity=capacity,
                ticket_price=ticket_price,
                n_simulations=n_simulations
            )
            
            scenario = OverbookingScenario(
                overbooking_level=overbooking,
                capacity=capacity,
                bookings=total_bookings,
                expected_shows=metrics['expected_shows'],
                expected_noshows=metrics['expected_noshows'],
                expected_denied=metrics['expected_denied'],
                expected_empty_seats=metrics['expected_empty_seats'],
                expected_revenue=metrics['expected_revenue'],
                expected_cost=metrics['expected_cost'],
                expected_profit=metrics['expected_profit'],
                probability_denied=metrics['probability_denied'],
                profit_std=metrics['profit_std']
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def analyze_capacity(self,
                         current_bookings: int,
                         noshow_prob: float,
                         capacity: int,
                         ticket_price: float) -> Dict[str, Any]:
        """
        Analyze current capacity situation.
        
        Args:
            current_bookings: Current number of bookings
            noshow_prob: Expected no-show probability
            capacity: Aircraft capacity
            ticket_price: Average ticket price
            
        Returns:
            Capacity analysis report
        """
        show_prob = 1.0 - noshow_prob
        expected_shows = current_bookings * show_prob
        expected_remaining = capacity - expected_shows
        utilization = expected_shows / capacity * 100 if capacity > 0 else 0
        
        # Risk of overflow is the fraction of simulations with more shows than seats.
        simulated_shows = self.simulate_shows(current_bookings, show_prob, 10000)
        overflow_risk = float((simulated_shows > capacity).mean())
        
        return {
            'current_bookings': current_bookings,
            'capacity': capacity,
            'expected_shows': round(expected_shows, 1),
            'expected_remaining_seats': round(max(0, expected_remaining), 1),
            'expected_utilization': round(utilization, 1),
            'overflow_risk': round(overflow_risk, 4),
            'recommendation': self._get_capacity_recommendation(
                current_bookings, expected_remaining, overflow_risk
            )
        }
    
    def _get_capacity_recommendation(self,
                                      bookings: int,
                                      remaining: float,
                                      overflow_risk: float) -> str:
        """Generate capacity recommendation."""
        if remaining < 0:
            return "Capacity exceeded. Consider upgrading aircraft or rebooking passengers."
        elif remaining < 5:
            return f"Almost full ({remaining:.0f} seats expected remaining). Limited flexibility."
        elif remaining < 20:
            return f"Moderate availability ({remaining:.0f} seats expected remaining)."
        else:
            return (
                f"Good availability ({remaining:.0f} seats expected remaining). "
                "No overbooking recommendation is made without validated costs and risk limits."
            )
    
    def get_recommendation(self) -> Dict[str, Any]:
        """
        Get general overbooking recommendation based on simulation results.
        
        This is a template method - actual recommendations depend on
        specific scenarios simulated.
        
        Returns:
            Recommendation dictionary with guidance for overbooking decisions
        """
        return {
            'safe_overbooking_level': None,
            'recommendation_available': False,
            'key_considerations': [
                'Historical no-show rates for the route',
                'Day of week and seasonality',
                'Passenger mix (business vs. leisure)',
                'Connection status and layover requirements',
                'Competition and rebooking costs'
            ],
            'disclaimer': (
                'No generic or safe overbooking level is recommended. Use scenario results '
                'only after validating input probabilities, costs, passenger protections, '
                'and operational risk limits.'
            )
        }
    
    def compare_scenarios(self,
                          scenarios: List[OverbookingScenario],
                          strategy: str = 'profit') -> Dict[str, Any]:
        """
        Compare different overbooking scenarios.
        
        Args:
            scenarios: List of simulated scenarios
            strategy: Optimization strategy ('profit', 'fill_rate', 'risk_adjusted')
            
        Returns:
            Comparison results and best scenario
        """
        if not scenarios:
            return {'error': 'No scenarios to compare'}
        
        # Calculate scores based on strategy
        scores = []
        for s in scenarios:
            if strategy == 'profit':
                score = s.expected_profit
            elif strategy == 'fill_rate':
                score = -s.expected_empty_seats  # Minimize empty seats
            elif strategy == 'risk_adjusted':
                # Risk-adjusted return: expected profit minus risk penalty
                risk_penalty = s.probability_denied * self.cost_params.denied_boarding_cost
                score = s.expected_profit - risk_penalty
            else:
                score = s.expected_profit
            
            scores.append({
                'overbooking_level': s.overbooking_level,
                'score': score,
                'expected_profit': s.expected_profit,
                'probability_denied': s.probability_denied
            })
        
        # Find best scenario
        best = max(scores, key=lambda x: x['score'])
        
        # Calculate relative improvement
        base_profit = scenarios[0].expected_profit
        improvement = (best['expected_profit'] - base_profit) / base_profit * 100 if base_profit > 0 else 0
        
        return {
            'strategy': strategy,
            'best_overbooking_level': best['overbooking_level'],
            'best_expected_profit': best['expected_profit'],
            'best_probability_denied': best['probability_denied'],
            'improvement_vs_baseline': f"{improvement:.1f}%",
            'operational_recommendation_available': False,
            'disclaimer': (
                'This is only the highest-scoring simulated scenario under the supplied '
                'inputs and cost assumptions; it is not an operational recommendation.'
            ),
            'all_scores': sorted(scores, key=lambda x: x['score'], reverse=True)[:5],
            'summary': f"The highest-scoring simulated level under the {strategy} strategy is "
                      f"{best['overbooking_level']} additional bookings, with modeled profit of "
                      f"${best['expected_profit']:.2f} and {best['probability_denied']:.2%} "
                      "simulated denied-boarding probability. This is not an operational recommendation."
        }


def run_demo():
    """Run demonstration of the overbooking simulator."""
    print("=" * 60)
    print("OVERBOOKING SIMULATOR DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Initialize simulator
    costs = CostParameters(
        compensation_cost=1500,
        empty_seat_cost=300,
        denied_boarding_cost=5000
    )
    simulator = OverbookingSimulator(cost_params=costs, random_seed=42)
    
    # Flight parameters
    capacity = 180  # Boeing 737 typical capacity
    current_bookings = 150
    avg_noshow_prob = 0.12  # 12% historical no-show rate
    ticket_price = 350
    
    print(f"Flight Parameters:")
    print(f"  Capacity: {capacity}")
    print(f"  Current Bookings: {current_bookings}")
    print(f"  Avg No-Show Rate: {avg_noshow_prob:.1%}")
    print(f"  Ticket Price: ${ticket_price}")
    print()
    
    # Analyze current capacity
    print("Current Capacity Analysis:")
    analysis = simulator.analyze_capacity(current_bookings, avg_noshow_prob, capacity, ticket_price)
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    print()
    
    # Run overbooking simulation
    print("Overbooking Simulation (Monte Carlo, 10,000 runs):")
    print("-" * 60)
    
    scenarios = simulator.simulate_scenario(
        capacity=capacity,
        bookings=current_bookings,
        avg_noshow_prob=avg_noshow_prob,
        ticket_price=ticket_price,
        max_overbooking=20,
        n_simulations=10000
    )
    
    # Show selected scenarios
    for s in scenarios[::5]:  # Every 5th scenario
        print(f"  +{s.overbooking_level:2d} bookings: Profit=${s.expected_profit:,.0f}, "
              f"Denied={s.expected_denied:.1f}, Empty={s.expected_empty_seats:.1f}")
    
    print()
    
    # Get recommendation
    comparison = simulator.compare_scenarios(scenarios, strategy='profit')
    print("Scenario Comparison (Profit Strategy):")
    print(f"  Best Level: +{comparison['best_overbooking_level']} bookings")
    print(f"  Expected Profit: ${comparison['best_expected_profit']:,.0f}")
    print(f"  Probability of Denied Boarding: {comparison['best_probability_denied']:.2%}")
    print()
    print(f"Summary: {comparison['summary']}")


if __name__ == "__main__":
    run_demo()