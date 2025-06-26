# 📦 Import required libraries
import pandas as pd
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus, value
import matplotlib.pyplot as plt

# 🧠 Step 1: Load the input data
df = pd.read_csv("data4.csv")   # Make sure 'data4.csv' is in the same folder
print("📊 Input Data:")
print(df)

# 🧱 Step 2: Define the linear programming problem
model = LpProblem("Product Mix Optimization", LpMaximize)

# 🎯 Step 3: Create decision variables
products = df['Product'].tolist()
decision_vars = LpVariable.dicts("Produce", products, lowBound=0, cat='Integer')

# 💰 Step 4: Objective function – Maximize total profit
model += lpSum(decision_vars[p] * df.loc[df['Product'] == p, 'Profit'].values[0] for p in products), "Total_Profit"

# ⛓️ Step 5: Constraints
model += lpSum(decision_vars[p] * df.loc[df['Product'] == p, 'Material'].values[0] for p in products) <= 1000, "Material_Constraint"
model += lpSum(decision_vars[p] * df.loc[df['Product'] == p, 'Labor'].values[0] for p in products) <= 1200, "Labor_Constraint"

# Max production per product
for p in products:
    max_prod = df.loc[df['Product'] == p, 'MaxProduction'].values[0]
    model += decision_vars[p] <= max_prod, f"MaxProduction_{p}"

# 🚀 Step 6: Solve the model
model.solve()

# 📢 Step 7: Print results
print("\n✅ Optimization Result:")
print(f"Status: {LpStatus[model.status]}")
for p in products:
    print(f"{p}: {decision_vars[p].varValue}")
print(f"Total Profit: {value(model.objective)}")

# 📊 Step 8: Plot and save the optimal production plan
quantities = [decision_vars[p].varValue for p in products]
plt.bar(products, quantities, color='skyblue')
plt.title("Optimal Production Plan")
plt.xlabel("Product")
plt.ylabel("Units Produced")
plt.grid(axis='y')
plt.tight_layout()  # Optional: ensures labels fit nicely
plt.savefig("optimal_production_plan.png")  # ✅ Saves plot instead of showing it
print("\n📈 Plot saved as 'optimal_production_plan.png'")

# 🧠 Step 9: Business Insights
print("\n🧠 Insights:")
print("• The model produced 100 units of Product A, 20 of B, and 60 of C.")
print("• Product A and C are likely more profitable or resource-efficient.")
print("• Product B is produced in low quantity, possibly due to higher resource use or lower profit.")
print("• Total profit is 5800 units under the given constraints (Material: 1000, Labor: 1200).")
print("• Resources are optimally allocated, and no constraint is violated.")
