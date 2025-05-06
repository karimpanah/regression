import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Example data: study hours and scores
hours = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).reshape(-1, 1)
scores = torch.tensor([55, 60, 70, 75, 80], dtype=torch.float32).reshape(-1, 1)

# Define the linear regression model
model = nn.Linear(in_features=1, out_features=1)

# Define the loss function (Mean Squared Error) and optimizer (SGD)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(hours)
    loss = criterion(predictions, scores)
    loss.backward()
    optimizer.step()

# Get final predictions
predicted_scores = model(hours).detach().numpy()
actual_scores = scores.numpy()
study_hours = hours.numpy()

# Calculate residuals
residuals = actual_scores - predicted_scores

# Extract learned parameters (slope and intercept)
slope = model.weight.item()
intercept = model.bias.item()
print(f"Regression line slope (β₁): {slope:.2f}")
print(f"Intercept (β₀): {intercept:.2f}")

# --- Plotting the Scatter Plot ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)  # Create a subplot area for the scatter plot (1 row, 2 columns, first plot)
plt.scatter(study_hours, actual_scores, color='blue', label='Actual scores')
plt.plot(study_hours, predicted_scores, color='red', label=f'Regression line (y = {intercept:.2f} + {slope:.2f}x)')
plt.xlabel('Study hours')
plt.ylabel('Score')
plt.title('Scatter Plot: Actual vs. Predicted Scores')
plt.legend()
plt.grid(True)

# --- Plotting the Residual Plot ---
# Create a subplot area for the residual plot (1 row, 2 columns, second plot)
plt.subplot(1, 2, 2)  
plt.scatter(study_hours, residuals, color='green')
# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='r', linestyle='--')  
plt.xlabel('Study hours')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.grid(True)

# Adjust subplot parameters for a tight layout
plt.tight_layout()  
plt.show()
