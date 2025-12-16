"""
Multivariable Linear Regression Project
Assignment 6 Part 3

Group Members:
- 
- 
- 
- 

Dataset: [Name of your dataset]
Predicting: [What you're predicting]
Features: [List your features]
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def load_and_explore_data(filename):
    """
    Load the car price data and explore it
    
    Args:
        filename: name of the CSV file to load
    
    Returns:
        pandas DataFrame containing the data
    """
    data = pd.read_csv(filename)
    
    print("=== Car Price Data ===")
    print(f"\nFirst 5 rows:")
    print(data.head())
    
    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    
    print(f"\nBasic statistics:")
    print(data.describe())
    
    print(f"\nColumn names: {list(data.columns)}")
    
    return data


def visualize_features(data):
    """
    Create scatter plots for each feature vs Price
    
    Args:
        data: pandas DataFrame with features and Price
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Car Features vs Price', fontsize=16, fontweight='bold')
    
    # Plot 1: Mileage vs Price
    axes[0, 0].scatter(data['Mileage'], data['Price'], color='blue', alpha=0.6)
    axes[0, 0].set_xlabel('Mileage (1000s of miles)')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].set_title('Mileage vs Price')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Age vs Price
    axes[0, 1].scatter(data['Age'], data['Price'], color='green', alpha=0.6)
    axes[0, 1].set_xlabel('Age (years)')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].set_title('Age vs Price')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Brand vs Price
    axes[1, 0].scatter(data['Brand'], data['Price'], color='red', alpha=0.6)
    axes[1, 0].set_xlabel('Brand (0=Toyota, 1=Honda, 2=Ford)')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].set_title('Brand vs Price')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Leave empty for now (or add another feature later)
    axes[1, 1].text(0.5, 0.5, 'Space for additional features', 
                    ha='center', va='center', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('car_features.png', dpi=300, bbox_inches='tight')
    print("\n✓ Feature plots saved as 'car_features.png'")
    plt.show()


def prepare_features(data):
    """
    Separate features (X) from target (y)
    
    Args:
        data: pandas DataFrame with all columns
    
    Returns:
        X - DataFrame with feature columns
        y - Series with target column
    """
    # Select multiple feature columns
    feature_columns = ['Mileage', 'Age', 'Brand']
    X = data[feature_columns]
    y = data['Price']
    
    print(f"\n=== Feature Preparation ===")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"\nFeature columns: {list(X.columns)}")
    
    return X, y


def split_data(X, y):
    """
    Split data into train/test 
    
    NOTE: We're splitting differently than usual to match our unplugged activity!
    First 15 cars = training, Last 3 cars = testing (just like you did manually)
    
    Also NOTE: We're NOT scaling features in this example so the coefficients
    are easy to interpret and compare to your manual equation!
    
    Args:
        X: features DataFrame
        y: target Series
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Split to match unplugged activity: first 15 for training, last 3 for testing
    # Note: For assignment, you should be using the train_test_split function
    X_train = X.iloc[:15]  # First 15 rows
    X_test = X.iloc[15:]   # Remaining rows (should be 3)
    y_train = y.iloc[:15]
    y_test = y.iloc[15:]
    
    print(f"\n=== Data Split (Matching Unplugged Activity) ===")
    print(f"Training set: {len(X_train)} samples (first 15 cars)")
    print(f"Testing set: {len(X_test)} samples (last 3 cars - your holdout set!)")
    print(f"\nNOTE: We're NOT scaling features here so coefficients are easy to interpret!")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, feature_names):
    """
    Train a multivariable linear regression model
    
    Args:
        X_train: training features (scaled)
        y_train: training target values
        feature_names: list of feature column names
    
    Returns:
        trained LinearRegression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"\n=== Model Training Complete ===")
    print(f"Intercept: ${model.intercept_:.2f}")
    print(f"\nCoefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:.2f}")
    
    print(f"\nEquation:")
    equation = f"Price = "
    for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
        if i == 0:
            equation += f"{coef:.2f} × {name}"
        else:
            equation += f" + ({coef:.2f}) × {name}"
    equation += f" + {model.intercept_:.2f}"
    print(equation)
    
    return model


def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate the model's performance
    
    Args:
        model: trained model
        X_test: testing features
        y_test: testing target values
        feature_names: list of feature names
    
    Returns:
        predictions array
    """
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print(f"\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"  → Model explains {r2*100:.2f}% of price variation")
    
    print(f"\nRoot Mean Squared Error: ${rmse:.2f}")
    print(f"  → On average, predictions are off by ${rmse:.2f}")
    
    # Feature importance (absolute value of coefficients)
    print(f"\n=== Feature Importance ===")
    feature_importance = list(zip(feature_names, np.abs(model.coef_)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, importance) in enumerate(feature_importance, 1):
        print(f"{i}. {name}: {importance:.2f}")
    
    return predictions


def compare_predictions(y_test, predictions, num_examples=5):
    """
    Show side-by-side comparison of actual vs predicted prices
    
    Args:
        y_test: actual prices
        predictions: predicted prices
        num_examples: number of examples to show
    """
    print(f"\n=== Prediction Examples ===")
    print(f"{'Actual Price':<15} {'Predicted Price':<18} {'Error':<12} {'% Error'}")
    print("-" * 60)
    
    for i in range(min(num_examples, len(y_test))):
        actual = y_test.iloc[i]
        predicted = predictions[i]
        error = actual - predicted
        pct_error = (abs(error) / actual) * 100
        
        print(f"${actual:>13.2f}   ${predicted:>13.2f}   ${error:>10.2f}   {pct_error:>6.2f}%")

def make_prediction(model, mileage, age, brand):
    """
    Make a prediction for a specific car
    
    Args:
        model: trained LinearRegression model
        mileage: mileage value (in thousands)
        age: age in years
        brand: brand code (0=Toyota, 1=Honda, 2=Ford)
    
    Returns:
        predicted price
    """
    # Create input array in the correct order: [Mileage, Age, Brand]
    car_features = pd.DataFrame([[mileage, age, brand]], 
                                 columns=['Mileage', 'Age', 'Brand'])
    predicted_price = model.predict(car_features)[0]
    
    brand_name = ['Toyota', 'Honda', 'Ford'][brand]
    
    print(f"\n=== New Prediction ===")
    print(f"Car specs: {mileage:.0f}k miles, {age} years old, {brand_name}")
    print(f"Predicted price: ${predicted_price:,.2f}")
    
    return predicted_price



if __name__ == "__main__":
    print("=" * 70)
    print("CAR PRICE PREDICTION - MULTIVARIABLE LINEAR REGRESSION")
    print("=" * 70)
    
    # Step 1: Load and explore
    data = load_and_explore_data('car_prices.csv')
    
    # Step 2: Visualize all features
    visualize_features(data)
    
    # Step 3: Prepare features
    X, y = prepare_features(data)
    
    # Step 4: Split data (no scaling for this example!)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 5: Train model
    model = train_model(X_train, y_train, X.columns)
    
    # Step 6: Evaluate model
    predictions = evaluate_model(model, X_test, y_test, X.columns)
    
    # Step 7: Compare predictions
    compare_predictions(y_test, predictions)

    # Step 8: Make a new prediction
    make_prediction(model, 45, 3, 0)  # 45k miles, 3 years, Toyota
    
    print("\n" + "=" * 70)
    print("✓ Example complete! Check out the saved plots.")
    print("=" * 70)
    - Identify which features look most important
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
    """
    print("\n" + "=" * 70)
    print("VISUALIZING RELATIONSHIPS")
    print("=" * 70)
    
    # Your code here
    # Hint: Use subplots like in Part 2!
    
    pass


def prepare_and_split_data(data):
    """
    Prepare X and y, then split into train/test
    
    TODO:
    - Separate features (X) and target (y)
    - Split into train/test (80/20)
    - Print the sizes
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print("PREPARING AND SPLITTING DATA")
    print("=" * 70)
    
    # Your code here
    
    pass


def train_model(X_train, y_train):
    """
    Train the linear regression model
    
    TODO:
    - Create and train a LinearRegression model
    - Print the equation with all coefficients
    - Print feature importance (rank features by coefficient magnitude)
    
    Args:
        X_train: training features
        y_train: training target
        feature_names: list of feature names
        
    Returns:
        trained model
    """
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    # Your code here
    
    pass


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    TODO:
    - Make predictions on test set
    - Calculate R² score
    - Calculate RMSE
    - Print results clearly
    - Create a comparison table (first 10 examples)
    
    Args:
        model: trained model
        X_test: test features
        y_test: test target
        
    Returns:
        predictions
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)
    
    # Your code here
    
    pass


def make_prediction(model):
    """
    Make a prediction for a new example
    
    TODO:
    - Create a sample input (you choose the values!)
    - Make a prediction
    - Print the input values and predicted output
    
    Args:
        model: trained model
        feature_names: list of feature names
    """
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    
    # Your code here
    # Example: If predicting house price with [sqft, bedrooms, bathrooms]
    # sample = pd.DataFrame([[2000, 3, 2]], columns=feature_names)
    
    pass


if __name__ == "__main__":
    # Step 1: Load and explore
    data = load_and_explore_data(DATA_FILE)
    
    # Step 2: Visualize
    visualize_data(data)
    
    # Step 3: Prepare and split
    X_train, X_test, y_train, y_test = prepare_and_split_data(data)
    
    # Step 4: Train
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate
    predictions = evaluate_model(model, X_test, y_test)
    
    # Step 6: Make a prediction, add features as an argument
    make_prediction(model)
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Analyze your results")
    print("2. Try improving your model (add/remove features)")
    print("3. Create your presentation")
    print("4. Practice presenting with your group!")

