import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, accuracy_score
from sklearn.preprocessing import StandardScaler as scaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import pickle
import os


# Read the training data from CSV files
training_raw_df = pd.read_csv('target_train.csv')
pharmacy_raw_df = pd.read_csv('rxclms_train.csv')
med_raw_df = pd.read_csv('medclms_train.csv')


# Pharmacy Data Processing


# Make values in the pharmacy dataframe numeric
pharmacy_raw_df.columns
value_mapping = {"N": 0, "Y": 1}
pharmacy_raw_df = pharmacy_raw_df.replace(value_mapping)

# Extract the first part of 'therapy_id' to create a new 'id' column
pharmacy_raw_df['id'] = pharmacy_raw_df['therapy_id'].str.split('-').str[0]

# Fill NaN values with 0 and drop 'therapy_id' column (now redundant)
pharmacy_raw_df = pharmacy_raw_df.fillna(0)
pharmacy_raw_df = pharmacy_raw_df.drop(columns=['therapy_id'])

# Make values in the 'maint_ind' column numeric
value_mapping = {"NONMAINT": 0, "MAINT": 1}
pharmacy_raw_df = pharmacy_raw_df.replace(value_mapping)

# Define columns of interest in the pharmacy dataframe
pharmacy_columns = ['id', "pay_day_supply_cnt", "rx_cost", "tot_drug_cost_accum_amt", "reversal_ind", "mail_order_ind",
                    "generic_ind", "maint_ind", "ddi_ind", "anticoag_ind", "diarrhea_treat_ind", "nausea_treat_ind", "seizure_treat_ind"]

# Group by 'id' and aggregate pharmacy data
pharmacy_df = pharmacy_raw_df[pharmacy_columns].groupby('id').agg({
    'pay_day_supply_cnt': 'mean',
    'rx_cost': 'mean',
    'tot_drug_cost_accum_amt': 'sum',
    'reversal_ind': 'max',
    'mail_order_ind': 'max',
    'generic_ind': 'max',
    'maint_ind': 'max',
    'ddi_ind': 'max',
    'anticoag_ind': 'max',
    'diarrhea_treat_ind': 'max',
    'nausea_treat_ind': 'max',
    'seizure_treat_ind': 'max'
})

# Reset index of dataframe and convert 'id' column to string (for a table join)
pharmacy_df.reset_index(inplace=True)
pharmacy_df['id'] = pharmacy_df['id'].astype(str)
training_raw_df['id'] = training_raw_df['id'].astype(str)

# Make values in the 'generic' column numeric
value_mapping = {"GENERIC": 0, "BRANDED": 1}
pharmacy_df = pharmacy_df.replace(value_mapping)

# Merge pharmacy data with training data
therapy_pharm = training_raw_df.merge(pharmacy_df, on='id', how='left')
therapy_pharm


# Medical Data Processing


# Extract the first part of 'therapy_id' to create a new 'id' column and drop 'therapy_id' (now redundant)
med_raw_df['id'] = med_raw_df['therapy_id'].str.split('-').str[0]
med_raw_df = med_raw_df.drop(columns=['therapy_id'])

# Define columns related to diagnosis
med_raw_df.columns
diag_columns = ['id', 'primary_diag_cd', 'diag_cd2', 'diag_cd3', 'diag_cd4', 'diag_cd5', 'diag_cd6', 'diag_cd7',
                'diag_cd8', 'diag_cd9']
diag_columns_wid = ['primary_diag_cd', 'diag_cd2', 'diag_cd3', 'diag_cd4', 'diag_cd5', 'diag_cd6', 'diag_cd7',
                    'diag_cd8', 'diag_cd9']

# Extract first letters of diagnosis codes in diagnosis columns (these letters correspond to a category of diagnosis from the ICD-10 guide)
med_diag = med_raw_df[diag_columns]
med_diag[diag_columns_wid] = med_diag[diag_columns_wid].apply(lambda x: x.str[0])
med_diag

# Update original dataframe with modified diagnosis columns (keeping only the diagnosis code letter)
med_raw_df[diag_columns] = med_diag

# Define columns of interest in the medical dataframe
med_columns = ['id', 'primary_diag_cd', 'diag_cd2', 'diag_cd3', 'diag_cd4', 'diag_cd5', 'diag_cd6', 'diag_cd7',
               'diag_cd8', 'diag_cd9', 'ade_diagnosis', 'seizure_diagnosis', 'pain_diagnosis', 'fatigue_diagnosis',
               'nausea_diagnosis', 'hyperglycemia_diagnosis', 'constipation_diagnosis', 'diarrhea_diagnosis']

# Group by 'id' and aggregate medical data
med_df = med_raw_df[med_columns].groupby('id').agg({
    'primary_diag_cd': 'sum',
    'diag_cd2': 'sum',
    'diag_cd3': 'sum',
    'diag_cd4': 'sum',
    'diag_cd5': 'sum',
    'diag_cd6': 'sum',
    'diag_cd7': 'sum',
    'diag_cd8': 'sum',
    'diag_cd9': 'sum',
    'ade_diagnosis': 'max',
    'seizure_diagnosis': 'max',
    'pain_diagnosis': 'max',
    'fatigue_diagnosis': 'max',
    'nausea_diagnosis': 'max',
    'hyperglycemia_diagnosis': 'max',
    'constipation_diagnosis': 'max',
    'diarrhea_diagnosis': 'max'
})

# Reset index of dataframe and convert 'id' to string (for a table join)
med_df.reset_index(inplace=True)
pharmacy_df['id'] = pharmacy_df['id'].astype(str)
med_df['id'] = med_df['id'].astype(str)

# Keep only unique diagnosis code letters
med_df[diag_columns_wid] = med_df[diag_columns_wid].apply(lambda x: x.apply(lambda y: set(str(y))))
med_df

# Extract letters from diagnosis columns
diag_letters = med_df[diag_columns_wid]
diag_letters


# Processing Diagnoses Letters and Concatenation


# Copy the DataFrame to avoid overwriting
df = diag_letters.copy()

# Create columns for each letter of the alphabet
letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# Set initial values for each letter's column
for letter in letters:
    df[letter] = 0

# Count the number of times each patient recieved a diagnosis code category (unique letter) across all doctors appointments
for letter in letters:
    for column in diag_letters.columns:
        df[letter] = diag_letters.apply(lambda row: any(letter in cell for cell in row), axis=1).astype(int)

# Drop the original diagnosis columns (now redundant)
df = df.drop(columns=diag_columns_wid)
print(df)

# Concatenate this count with the aggregated medical data dataframe
med_df = pd.concat([med_df, df], axis=1)
med_df.drop(columns=diag_columns_wid)


# Merging and Preprocessing


# Merge aggregated pharmacy data + patient demographic info with newly aggregated medical data
therapy_pharm_med = therapy_pharm.merge(med_df, on='id', how='left')
raw_df = therapy_pharm_med.copy()

# One-hot encode sex
raw_df = pd.get_dummies(raw_df, columns=['sex_cd'], prefix='sex')

# Count duplicates based on 'id'
dedup_df = raw_df.copy()
duplicate_counts = dedup_df['id'].value_counts()
print(duplicate_counts)


# Dealing with missing values


# Copy the DataFrame to avoid modifying the original
fill_na = raw_df.copy()

# Fill na values with nan
value_mapping = {np.nan: np.nan, 0.0: "False", 1.0: "True"}
fill_na[["cms_disabled_ind","cms_low_income_ind"]] = fill_na[["cms_disabled_ind","cms_low_income_ind"]].replace(value_mapping)
fill_na.fillna(np.nan, inplace=True)
fill_na

# Fill missing values for 'est_age' with mean
fill_na['est_age'].fillna(fill_na['est_age'].mean(), inplace=True)

# Fill categorical variables with mode
columns_to_fill_mode = ['cms_disabled_ind', 'cms_low_income_ind', 'sex_F', 'sex_M', 'ade_diagnosis', 'seizure_diagnosis',
                         'pain_diagnosis', 'fatigue_diagnosis', 'nausea_diagnosis', 'hyperglycemia_diagnosis',
                         'constipation_diagnosis', 'reversal_ind', 'mail_order_ind', 'generic_ind', 'maint_ind', 'ddi_ind',
                         'anticoag_ind', 'diarrhea_treat_ind', 'nausea_treat_ind', 'seizure_treat_ind', 'ade_diagnosis',
                         'seizure_diagnosis', 'diarrhea_diagnosis']

for column in columns_to_fill_mode:
    fill_na[column].fillna(fill_na[column].mode().iloc[0], inplace=True)

# Fill numerical variable 'race_cd' with mean
fill_na['race_cd'].fillna(fill_na['race_cd'].mean(), inplace=True)

# Function to fill remaining columns with mean
def fill_columns_with_mean(df):
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for letter in letters:
        column_name = letter
        if column_name in df.columns:
            column_mean = round(df[column_name].mean())
            df[column_name].fillna(column_mean, inplace=True)

fill_columns_with_mean(fill_na)

# Check for remaining NaN values
nan_count = fill_na.isna().sum()
print(nan_count)


# Data preprocessing


# Copy the DataFrame for further modifications
target_train = fill_na.copy()

# Map categorical values to 0 and 1
value_mapping = {"False": 0, "True": 1}
target_train = target_train.replace(value_mapping)

# Map binary values to 0 and 1
value_mapping = {False: 0, True: 1}
s = ['sex_F','sex_M']
target_train[s] = target_train[s].replace(value_mapping)

# Drop unnecessary columns
columns_to_remove = ['tgt_ade_dc_ind','id','therapy_id','therapy_start_date','therapy_end_date']
target_train = target_train.drop(columns=diag_columns_wid)

# Create binary columns for race
modified_tt = target_train.copy()
race_key = {
    0: 'Unknown',
    1: 'White',
    2: 'Black',
    3: 'Other',
    4: 'Asian',
    5: 'Hispanic',
    6: 'Native American'
}

# Create a binary column for each race, and set them all to 0 initially
for index, race in race_key.items():
    modified_tt[race] = 0

# Iterate through the DataFrame and set the appropriate race column to 1 based on 'race_cd'
for index, row in modified_tt.iterrows():
    race_cd = row['race_cd']
    if race_cd in race_key:
        modified_tt.at[index, race_key[race_cd]] = 1
    else:
        modified_tt.at[index, 'Unknown'] = 1

# Drop the original 'race_cd' column
modified_tt = modified_tt.drop(columns=['race_cd'])

# Update the target_train DataFrame
target_train = modified_tt.copy()
print(target_train)

# Log-transform numerical columns, adding a constant on the same scale the values are on to prevent log(0)
target_train['rx_cost'] = np.log10(target_train['rx_cost'] + 10)
target_train['tot_drug_cost_accum_amt'] = np.log10(target_train['tot_drug_cost_accum_amt'] + 1000)


# Use model on testing data


# Select features
X = target_train.drop(columns=['tgt_ade_dc_ind', 'id', 'therapy_id', 'therapy_start_date', 'therapy_end_date'])
# Select labels
y = target_train['tgt_ade_dc_ind']

# Fill missing values in specific columns
X['pay_day_supply_cnt'].fillna(X['pay_day_supply_cnt'].mean(), inplace=True)
X['rx_cost'].fillna(X['rx_cost'].mean(), inplace=True)
X['tot_drug_cost_accum_amt'].fillna(X['tot_drug_cost_accum_amt'].mean(), inplace=True)

# Drop any remaining rows with NaN values
X = X.dropna()

nan_count = X.isna().sum()
print(nan_count)


# Feature selection


# Apply SelectKBest to extract top 11 features most associated with our outcome using chi-squared test
bestfeatures = SelectKBest(score_func=chi2, k=11)
fit = bestfeatures.fit(X, y)

# Get feature scores and names
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']

# Print 11 best features
print(featureScores.nlargest(11, 'Score'))

# Select top 11 features
features = featureScores.nlargest(11, 'Score')
cols = features['Specs']
X = X[cols]
X.columns


# Prediction on New Data


# Load the model for alpha=0.01
model_dir = 'saved_models'
model_filename = os.path.join(model_dir, 'model_alpha_0.01.keras')
loaded_model = keras.models.load_model(model_filename)

# Make predictions on the new data using the loaded model
data = pd.read_csv("test_data.csv", index_col = False)
data = data.drop(columns='Unnamed: 0')
data.drop(columns=['id', 'therapy_id'])
final_data = data.drop(columns=['id', 'therapy_id'])

y_pred_new = loaded_model.predict(final_data)
y_pred_prob_flat = y_pred_new.flatten()


# Save results


# Create a DataFrame for results
results_df = pd.DataFrame({
    'ID': data['id'],
    'SCORE': y_pred_prob_flat
})
#Most likely to drop out of treatment are listed first (highest probability)
results_df = results_df.sort_values(by='SCORE', ascending=False)
results_df['RANK'] = range(1, len(results_df) + 1)
print(results_df)
results_df.style.format("{:.f}")

# Save results to CSV
results_df.to_csv("results.csv", index=False)

# Save binary results to CSV for additional analysis or visualization
results_df_graphs = pd.DataFrame({
    'id': data['id'],
    'SCORE': (y_pred_new > 0.5).astype(int).flatten()
})
results_df_graphs.to_csv("results_df_graphs.csv", index=False)


# Neural Network Model (loaded previously but infrastructure given here)


# Define a directory to save the models
model_dir = 'saved_models'
os.makedirs(model_dir, exist_ok=True)

# Initialize variables to store ROC curve data for alpha=0.01
fpr_alpha_0_01, tpr_alpha_0_01, thresholds_alpha_0_01 = None, None, None

# Load data
X = target_train.drop(columns=['tgt_ade_dc_ind', 'id', 'therapy_id', 'therapy_start_date', 'therapy_end_date'])
y = target_train['tgt_ade_dc_ind']

cols = features['Specs']
X = X[cols]

#80/20 train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a range of L2 regularization strengths (alphas) to test
alphas = np.logspace(-5, 2, 8)

# Create empty lists to store ROC AUC values for each alpha
roc_auc_values = []

# Class weights for imbalanced data
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Iterate over different L2 regularization strengths
for alpha in alphas:

    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(alpha)),
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(alpha)),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the neural network
    model_final = model.fit(X_train, y_train, epochs=15, batch_size=64, verbose=0,
                            validation_data=(X_val, y_val), class_weight=class_weight_dict)

    # Make predictions on the validation set
    y_pred = model.predict(X_val)

    # Calculate and store ROC AUC
    auc = roc_auc_score(y_val, y_pred)
    roc_auc_values.append(auc)

    if alpha == 0.01:
        # Calculate the fpr and tpr for the ROC curve
        y_val_final = y_val
        y_pred_final = y_pred
        fpr_alpha_0_01, tpr_alpha_0_01, thresholds_alpha_0_01 = roc_curve(y_val, y_pred)

        # Save the model to a file
        model_filename = os.path.join(model_dir, 'model_alpha_0.01.keras')
        model.save(model_filename)

        # Save other relevant data
        model_data = {
            'alphas': alphas,
            'roc_auc_values': roc_auc_values,
            'fpr_alpha_0_01': fpr_alpha_0_01,
            'tpr_alpha_0_01': tpr_alpha_0_01,
            'thresholds_alpha_0_01': thresholds_alpha_0_01,
        }
        #Write the file
        model_data_filename = os.path.join(model_dir, 'model_data_alpha_0.01.keras')
        with open(model_data_filename, 'wb') as file:
            pickle.dump(model_data, file)

        # Print AUC and classification report
        print(f"AUC (alpha={alpha}): {auc}")
        y_pred_binary = (y_pred > 0.5).astype(int)
        print(classification_report(y_val, y_pred_binary))
        
        # Create a confusion matrix
        conf_matrix = confusion_matrix(y_val, y_pred_binary)
        
        # Visualize the confusion matrix
        labels = ['Negative', 'Positive']  # Assuming binary classification (0 and 1)
        sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        # Calculate percentages from the confusion matrix
        tn, fp, fn, tp = conf_matrix.ravel()
        total_samples = tn + fp + fn + tp

        percentages = {
            'True Positive': (tp / total_samples) * 100,
            'False Positive': (fp / total_samples) * 100,
            'True Negative': (tn / total_samples) * 100,
            'False Negative': (fn / total_samples) * 100
        }

        print("Percentages:")
        for label, percentage in percentages.items():
            print(f"{label}: {percentage:.2f}%")

        if alpha == 0.01:
            # Save confusion matrix for alpha=0.01
            conf_matrix_alpha_0_01 = conf_matrix

# Plot the ROC AUC values
plt.plot(alphas, roc_auc_values, linestyle='-')
plt.xlabel("L2 Regularization Strength (Alpha)")
plt.ylabel("ROC AUC")
plt.title("ROC AUC vs. L2 Regularization Strength")
plt.xscale("log")
plt.grid()

# Show the plot for alpha=0.01
if fpr_alpha_0_01 is not None and tpr_alpha_0_01 is not None:
    plt.figure()
    plt.plot(fpr_alpha_0_01, tpr_alpha_0_01)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (alpha=0.01)')
    plt.show()

# Plot the ROC curve for alpha=0.01
if fpr_alpha_0_01 is not None and tpr_alpha_0_01 is not None:
    plt.figure()
    plt.plot(fpr_alpha_0_01, tpr_alpha_0_01, label='ROC Curve (alpha=0.01, AUC=0.9376)')
    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


# Prediction on New Data


# Read in new data and standardize by column
final_data = data.drop(columns=['id', 'therapy_id'])
final_data = final_data[cols]
final_data = scaler.transform(final_data)

# Define the path to the directory where the model is saved
model_dir = 'saved_models'

# Load the model for alpha=0.01
model_filename = os.path.join(model_dir, 'model_alpha_0.01.keras')
loaded_model = keras.models.load_model(model_filename)

y_pred_new = loaded_model.predict(final_data)
y_pred_prob_flat = y_pred_new.flatten()


# Save results


# Create a DataFrame for results
results_df = pd.DataFrame({
    'ID': data['id'],
    'SCORE': y_pred_prob_flat
})
results_df = results_df.sort_values(by='SCORE', ascending=False)
results_df['RANK'] = range(1, len(results_df) + 1)
print(results_df)
results_df.style.format("{:.f}")

# Save results to CSV
results_df.to_csv("results.csv", index=False)

# Save binary results to CSV for additional analysis or visualization
results_df_graphs = pd.DataFrame({
    'id': data['id'],
    'SCORE': (y_pred_new > 0.5).astype(int).flatten()
})
results_df_graphs.to_csv("results_df_graphs.csv", index=False)