import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('soil_fertility_rf_model.pkl')

# Define the feature names used during training
feature_names = ['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']

# Function to make predictions
def predict_soil_fertility(features):
    df = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(df)
    return prediction[0]

# Streamlit app
st.title('Soil Fertility Prediction')

# Input fields for features
features = []
for feature in feature_names:
    value = st.number_input(f'Enter {feature}')
    features.append(value)

# Predict button
if st.button('Predict'):
    prediction = predict_soil_fertility(features)
    st.write(f'Predicted Soil Fertility: {prediction}')


"""import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

# Load the trained model
model = joblib.load('soil_fertility_rf_model.pkl')

# Define the feature names used during training
feature_names = ['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']

# Function to make predictions
def predict_soil_fertility(features):
    if len(features) != len(feature_names):
        raise ValueError(f'Input features must have {len(feature_names)} elements, but got {len(features)}')

    df = pd.DataFrame([features], columns=feature_names)
    
    # Standardize features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Apply PCA
    pca = PCA(n_components=min(df_scaled.shape[0], df_scaled.shape[1]))  # Use min(n_samples, n_features)
    df_pca = pca.fit_transform(df_scaled)
    
    # Predict
    prediction = model.predict(df_pca)
    return prediction[0]

# Function to apply PCA and KDE
def apply_pca_and_kde(data):
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Apply PCA
    n_components = min(data_scaled.shape[0], data_scaled.shape[1])
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)

    # Apply KDE
    kde = gaussian_kde(data_pca.T)
    return kde, data_pca

# Streamlit app
st.title('Soil Fertility Prediction')

# Input fields for features
features = []
st.sidebar.header('Input Features')
for feature in feature_names:
    value = st.sidebar.number_input(f'Enter {feature}', value=0.0)
    features.append(value)

# Display feature histograms
st.sidebar.header('Feature Histograms')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=pd.DataFrame([features], columns=feature_names), bins=10, kde=True, ax=ax)
ax.set_title('Feature Distribution')
st.pyplot(fig)  # Use st.pyplot to display plots in Streamlit

# Predict button
if st.button('Predict'):
    try:
        prediction = predict_soil_fertility(features)
        st.write(f'Predicted Soil Fertility: {prediction}')

        # Display PCA and KDE
        try:
            # Generate sample data for PCA and KDE (replace with actual data in practice)
            np.random.seed(42)
            sample_data = np.random.rand(100, len(feature_names))  # 100 samples with feature_names dimensions
            
            # Apply PCA and KDE
            kde, data_pca = apply_pca_and_kde(sample_data)

            # Plot KDE results
            x, y = np.meshgrid(np.linspace(data_pca[:, 0].min(), data_pca[:, 0].max(), 100),
                               np.linspace(data_pca[:, 1].min(), data_pca[:, 1].max(), 100))
            positions = np.vstack([x.ravel(), y.ravel()])
            density = kde(positions).reshape(x.shape)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.contourf(x, y, density, cmap='viridis')
            ax.set_title('KDE of PCA-transformed Data')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            st.pyplot(fig)  # Use st.pyplot to display plots in Streamlit

        except Exception as e:
            st.write('Error generating PCA and KDE plots:', e)

        # Display Feature Importance (if model has feature importances)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

            st.subheader('Feature Importance')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importances, ax=ax)
            ax.set_title('Feature Importance')
            st.pyplot(fig)  # Use st.pyplot to display plots in Streamlit

    except ValueError as ve:
        st.write(f'Error: {ve}')
    except Exception as e:
        st.write(f'Error during prediction: {e}')"""
