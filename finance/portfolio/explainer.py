import os
import logging
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioExplainer:
    def __init__(self, model: str = "gemini-1.5-flash-latest"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not found in environment.")
            self.llm = None
        else:
            self.llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
        
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert financial consultant for DeepChain. 
        Your task is to explain a personalized portfolio strategy to a client in plain, encouraging English.
        The client's strategy was derived dynamically using their profile and live market data.

        --- CLIENT PROFILE SUMMARY ---
        Age: {age}
        Dependents: {dependents}
        Investment Horizon: {horizon}
        Primary Goal: {goal}
        Risk Profile (Calculated): {risk_profile}
        Net Investable Amount: ₹{investable_amount}
        
        --- FINANCIAL HEALTH STATUS ---
        Emergency Fund Shortfall: ₹{ef_shortfall}
        High Interest Debt: ₹{debt}
        Insurance Status: {insurance}
        
        --- PROPOSED ALLOCATIONS ---
        {allocations}

        --- INSTRUCTIONS ---
        1. Contextualize the strategy: Mention how their {dependents} dependents, {horizon} horizon, and goal of '{goal}' influenced their {risk_profile} risk rating.
        2. Financial Health Priority: If there's an EF shortfall or high-interest debt, explain why the system is prioritizing those first before market exposure.
        3. Sector Rationale: Explain WHY these specific sectors were chosen for the remaining funds (mention live market trends).
        4. Keep the tone professional, empathetic, and expert.
        5. Do NOT change any numbers or percentages.

        Professional Explanation:
        """)

    def explain(self, profile: Dict[str, Any], results: Dict[str, Any]) -> str:
        if not self.llm:
            return "LLM Explainer not available. Please check GOOGLE_API_KEY."

        # Format allocations for the prompt
        alloc_str = ""
        for sector, data in results["allocations"].items():
            alloc_str += f"- {sector}: {data['weight']}% (₹{data['amount']})\n"
        
        if not alloc_str:
            alloc_str = "No allocations recommended due to health check priorities."

        try:
            chain = self.prompt_template | self.llm
            response = chain.invoke({
                "age": profile.get("age"),
                "dependents": profile.get("dependents"),
                "horizon": profile.get("investment_horizon"),
                "goal": profile.get("primary_goal"),
                "risk_profile": results["risk_profile"],
                "investable_amount": results["surplus_data"]["net_investable_now"],
                "ef_shortfall": results["health_status"]["ef_shortfall"],
                "debt": results["health_status"]["high_interest_debt"],
                "insurance": "Missing" if results["health_status"]["insurance_missing"] else "Covered",
                "allocations": alloc_str
            })
            return response.content
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return f"Mathematical strategy generated, but explanation failed: {str(e)}"

if __name__ == "__main__":
    # Test Explainer
    explainer = PortfolioExplainer()
    mock_profile = {"age": 30}
    mock_results = {
        "risk_profile": "Moderate",
        "health_status": {"ef_shortfall": 0, "high_interest_debt": 0, "insurance_missing": False},
        "surplus_data": {"net_investable_now": 200000},
        "allocations": {"Nifty IT": {"weight": 30.0, "amount": 60000}}
    }
    print(explainer.explain(mock_profile, mock_results))
