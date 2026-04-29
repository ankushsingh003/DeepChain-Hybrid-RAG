import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioStrategy:
    def __init__(self):
        # Boundary Rules
        self.MIN_EF_MONTHS = 3
        self.HIGH_INTEREST_THRESHOLD = 12.0
        self.SINGLE_SECTOR_CAP = 0.30  # 30%
        self.MIN_SECTOR_ALLOCATION = 0.05  # 5%

    def calculate_allocation(self, profile: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the 4-step strategy logic.
        """
        logger.info("Starting Strategy Calculation...")
        
        # Step 1: Financial Health Check
        health_status = self._health_check(profile)
        
        # Step 2: Investable Surplus Calculation
        surplus_data = self._calculate_surplus(profile, health_status)
        
        # Step 3: Risk Profile Derivation
        risk_profile = self._derive_risk_profile(profile, health_status)
        
        # Step 4: Sector Allocation
        allocations = self._allocate_sectors(market_data, surplus_data, risk_profile)
        
        return {
            "health_status": health_status,
            "surplus_data": surplus_data,
            "risk_profile": risk_profile,
            "allocations": allocations
        }

    def _health_check(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        income = profile.get("monthly_income", 0) + \
                 profile.get("pension", 0) + \
                 profile.get("govt_allowances", 0) + \
                 profile.get("additional_income", 0)
        
        expenses = profile.get("monthly_expenses", 0)
        
        # Emergency Fund Check
        required_ef = expenses * self.MIN_EF_MONTHS
        current_ef = profile.get("existing_savings", 0) if profile.get("emergency_fund_exists") else 0
        ef_shortfall = max(0, required_ef - current_ef)
        
        # Debt Check
        liabilities = profile.get("liabilities", [])
        high_interest_debt = sum(l.get("amount", 0) for l in liabilities if l.get("interest_rate", 0) > self.HIGH_INTEREST_THRESHOLD)
        
        return {
            "monthly_surplus": income - expenses,
            "ef_shortfall": ef_shortfall,
            "high_interest_debt": high_interest_debt,
            "insurance_missing": not (profile.get("life_insurance") and profile.get("health_insurance")),
            "is_healthy": ef_shortfall == 0 and high_interest_debt == 0
        }

    def _calculate_surplus(self, profile: Dict[str, Any], health: Dict[str, Any]) -> Dict[str, Any]:
        monthly_surplus = health["monthly_surplus"]
        amount_to_invest_now = profile.get("amount_to_invest", 0)
        
        # Prioritize EF and Debt from the immediate amount
        net_investable_now = amount_to_invest_now - health["ef_shortfall"] - health["high_interest_debt"]
        net_investable_now = max(0, net_investable_now)
        
        return {
            "net_investable_now": net_investable_now,
            "monthly_surplus": monthly_surplus,
            "ef_allocation": min(amount_to_invest_now, health["ef_shortfall"]),
            "debt_allocation": min(max(0, amount_to_invest_now - health["ef_shortfall"]), health["high_interest_debt"])
        }

    def _derive_risk_profile(self, profile: Dict[str, Any], health: Dict[str, Any]) -> str:
        age = profile.get("age", 30)
        dependents = profile.get("dependents", 0)
        horizon = profile.get("investment_horizon", "5yr")
        goal = profile.get("primary_goal", "Wealth Creation")
        
        # Base Equity Cap: 100 - age
        equity_cap = 100 - age
        
        # Condition: More dependents -> more conservative
        equity_cap -= (dependents * 7) # Increased penalty per dependent
        
        # Condition: Health Overrides (EF, Debt, Insurance)
        if not health["is_healthy"]:
            equity_cap -= 25 # Heavier penalty for poor financial health
        if health["insurance_missing"]:
            equity_cap -= 10
            
        # Condition: Investment Horizon
        if horizon == "1yr": equity_cap -= 30
        elif horizon == "3yr": equity_cap -= 15
        elif horizon == "10yr+": equity_cap += 10
        
        # Condition: Primary Goal
        if goal == "Capital Preservation": equity_cap -= 20
        elif goal == "Emergency Fund": equity_cap -= 40
        elif goal == "Wealth Creation": equity_cap += 5
        
        # Final Risk Profile Mapping
        if equity_cap > 75: return "Aggressive"
        if equity_cap > 45: return "Moderate"
        return "Conservative"

    def _allocate_sectors(self, market_data: Dict[str, Any], surplus: Dict[str, Any], risk: str) -> Dict[str, Any]:
        if surplus["net_investable_now"] <= 0:
            return {}

        # Score sectors based on market_data (Price, PE, Momentum, Sentiment, Risk)
        sector_scores = {}
        for name, data in market_data.items():
            # Basic Scoring Logic: 
            # + Momentum, + Sentiment, - PE (relative), - Risk Flags
            score = 0
            score += data.get("momentum_3m", 0) * 0.3
            score += data.get("sentiment_score", 0.5) * 20
            
            # Risk Penalty
            risk_count = len(data.get("risk_flags", []))
            score -= (risk_count * 10)
            
            # PE Penalty (if too high compared to historical average - simplified here)
            pe = data.get("pe_ratio", 20)
            if pe != "N/A":
                if pe > 40: score -= 15
                elif pe < 15: score += 10
            
            sector_scores[name] = max(0, score)

        # Normalize and apply Single Sector Cap
        total_score = sum(sector_scores.values())
        if total_score == 0: return {}

        allocations = {}
        for name, score in sector_scores.items():
            weight = score / total_score
            # Apply Cap
            weight = min(weight, self.SINGLE_SECTOR_CAP)
            
            # Filter Minimum
            if weight >= self.MIN_SECTOR_ALLOCATION:
                allocations[name] = {
                    "weight": round(weight * 100, 2),
                    "amount": round(weight * surplus["net_investable_now"], 2)
                }

        return allocations

if __name__ == "__main__":
    # Test Strategy
    strat = PortfolioStrategy()
    mock_profile = {
        "age": 30, "monthly_income": 100000, "monthly_expenses": 40000,
        "dependents": 1, "existing_savings": 50000, "emergency_fund_exists": True,
        "amount_to_invest": 200000, "life_insurance": True, "health_insurance": True,
        "liabilities": [{"amount": 50000, "interest_rate": 15.0}]
    }
    mock_market = {
        "Nifty IT": {"pe_ratio": 25, "momentum_3m": 10, "sentiment_score": 0.8, "risk_flags": []},
        "Nifty Bank": {"pe_ratio": 15, "momentum_3m": 2, "sentiment_score": 0.4, "risk_flags": ["NBFC Risk"]}
    }
    result = strat.calculate_allocation(mock_profile, mock_market)
    import json
    print(json.dumps(result, indent=4))
