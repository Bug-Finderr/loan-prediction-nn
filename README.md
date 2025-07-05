# LendingClub Loan Default Prediction

A neural network-based binary classification system for predicting loan defaults using LendingClub historical data. The model achieves 80.3% accuracy with 82.0% precision on loan repayment prediction.

## Dataset Overview

- **Records**: 396,030 loan applications
- **Features**: 25 selected features (reduced from 27 original)
- **Target**: Binary classification (Fully Paid vs Charged Off)
- **Class Distribution**: 80.4% Fully Paid, 19.6% Charged Off

## Model Architecture

**Neural Network Configuration:**

- Input Layer: 25 features
- Hidden Layer 1: 128 neurons + LeakyReLU(0.01)
- Hidden Layer 2: 64 neurons + LeakyReLU(0.01)
- Output Layer: 1 neuron + Sigmoid activation
- Optimizer: Adam (lr=0.001)
- Loss: Binary Cross-Entropy

## Performance Metrics

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 80.33% |
| Precision | 82.02% |
| Recall    | 96.73% |
| F1-Score  | 88.77% |
| AUC-ROC   | 70.99% |

## Confusion Matrix

|                        | Predicted Fully Paid | Predicted Charged Off |
| ---------------------- | -------------------- | --------------------- |
| **Actual Fully Paid**  | 20,033               | 10,395                |
| **Actual Charged Off** | 2,311                | 48,626                |

**Observations**:

- The model correctly predicts the majority of loan repayment outcomes.
- There are 10,395 false positives (loans predicted as charged off but were fully paid).
- There are 2,311 false negatives (loans predicted as fully paid but were charged off).

## Project Structure

```text
src/
├── bin/model.h5        # Neural network model file
├── eda.ipynb           # Exploratory data analysis
└── train.py            # Model training script

data/
├── lending_club_loan_two.csv    # Raw dataset
└── processed/                   # Preprocessed data
    ├── train_data_scaled.csv
    ├── test_data_scaled.csv
    ├── feature_info.csv
    ├── scaler_params.csv
    └── metadata.json
```

## Data Processing Pipeline

1. **Missing Value Handling**: Median imputation for numerical features
2. **Feature Engineering**: Created 8 derived features from financial ratios and dates
3. **Categorical Encoding**: Label encoding for categorical variables
4. **Feature Selection**: Removed multicollinear features (correlation > 0.8)
5. **Standardization**: StandardScaler for neural network optimization

## Key Features

**Top Predictive Features:**

- grade_numeric: Loan grade (A=7 to G=1)
- mort_acc: Number of mortgage accounts
- annual_inc: Annual income
- emp_length_num: Employment length in years
- credit_history_length: Years of credit history

**Engineered Features:**

- loan_to_income_ratio
- debt_to_credit_ratio
- total_credit_lines
- credit_history_length

## Usage

### Setup

**Note**: Use Python 3.8 - 3.12 for TensorFlow compatibility.

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### Training

```bash
python src/train.py
```

### Exploratory Data Analysis

```bash
jupyter notebook src/eda.ipynb
```

## Model Files

- **Training Data**: `data/processed/train_data_scaled.csv`
- **Test Data**: `data/processed/test_data_scaled.csv`
- **Feature Metadata**: `data/processed/feature_info.csv`
- **Preprocessing Info**: `data/processed/metadata.json`

## Data Quality

- **Missing Values**: Handled 4 features with missing data
- **Multicollinearity**: Removed 4 highly correlated features
- **Class Imbalance**: 4:1 ratio (Fully Paid:Charged Off)

## Implementation Notes

- Stratified train-test split (80:20) maintains class distribution
- StandardScaler applied for feature normalization
- LeakyReLU activation prevents dead neurons
- Binary cross-entropy loss optimized for classification task

## Results Interpretation and Business Recommendations

### Results Interpretation

The model demonstrates strong predictive performance with an F1-score of 88.77%, indicating a balanced trade-off between precision and recall. The high recall (96.73%) ensures that most loan defaults are correctly identified, while the precision (82.02%) minimizes false positives.

### Business Recommendations

1. **Loan Approval Optimization**:

   - Use the model to assess borrower risk and prioritize applications with high repayment likelihood.
   - Reduce financial losses by identifying high-risk loans early.

2. **Portfolio Management**:

   - Monitor loan portfolios for potential defaults and adjust interest rates or repayment terms accordingly.

3. **Customer Segmentation**:

   - Segment borrowers based on risk profiles to offer tailored loan products and services.

4. **ROI Potential**:
   - The model can significantly reduce charge-offs, improving profitability and operational efficiency.

### Business Value

By using the model, LendingClub can:

- Enhance decision-making for loan approvals.
- Improve customer satisfaction through personalized offerings.
- Achieve better financial outcomes by minimizing defaults and optimizing loan portfolios.
