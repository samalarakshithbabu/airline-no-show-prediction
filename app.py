"""
Streamlit Dashboard for Airline No-Show Prediction System
Complete ML Pipeline Interface with Analytics, Prediction, and Business Tools
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from ml.data_processing import clean_data, encode_categorical, prepare_features, split_data
from ml.feature_engineering import engineer_features
from ml.training import ModelTrainer
from ml.evaluation import ModelEvaluator, ModelComparator, plot_confusion_matrix
from ml.overbooking_simulator import OverbookingSimulator, CostParameters
from ml.drift_detection import DriftDetector
from ml.pipeline import MLPipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Custom CSS for attractive styling
st.set_page_config(
    page_title="✈️ Airline No-Show Prediction System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #4a7ab8 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 20px rgba(30, 60, 114, 0.3);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #1e3c72;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        font-size: 1.4rem;
        font-weight: bold;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 60, 114, 0.4);
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #a8d5ae;
        border-radius: 10px;
        padding: 1.25rem;
        margin: 1rem 0;
        color: #155724;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1.25rem;
        margin: 1rem 0;
        color: #856404;
    }
    
    .error-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 1px solid #f5c6cb;
        border-radius: 10px;
        padding: 1.25rem;
        margin: 1rem 0;
        color: #721c24;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e7f3ff 0%, #d4e8ff 100%);
        border: 1px solid #b8daff;
        border-radius: 10px;
        padding: 1.25rem;
        margin: 1rem 0;
        color: #004085;
    }
    
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #1e3c72;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'trained' not in st.session_state:
    st.session_state['trained'] = False
if 'models' not in st.session_state:
    st.session_state['models'] = None
if 'X_test' not in st.session_state:
    st.session_state['X_test'] = None
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None
if 'trainer' not in st.session_state:
    st.session_state['trainer'] = None
if 'feature_names' not in st.session_state:
    st.session_state['feature_names'] = None
if 'best_model_name' not in st.session_state:
    st.session_state['best_model_name'] = None


def run_eda(df, container):
    """Run exploratory data analysis."""
    container.markdown("#### 📊 Dataset Overview")
    
    col1, col2, col3, col4 = container.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Features", f"{len(df.columns)}")
    col3.metric("No-Show Rate", f"{(df['no_show'].mean() * 100):.1f}%")
    col4.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    
    container.markdown("#### 📋 Data Types")
    dtype_df = pd.DataFrame({
        'Column': df.dtypes.index,
        'Type': df.dtypes.values
    })
    container.dataframe(dtype_df, use_container_width=True)
    
    container.markdown("#### 📈 Statistical Summary")
    container.dataframe(df.describe(), use_container_width=True)


def show_visualizations(df, container):
    """Show data visualizations."""
    container.markdown("#### 🎯 Target Distribution")
    
    col1, col2 = container.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#2ecc71', '#e74c3c']
        counts = df['no_show'].value_counts()
        labels = ['Show', 'No-Show']
        ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, explode=[0, 0.05])
        ax.set_title('Passenger Show/No-Show Distribution')
        container.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        if 'booking_lead_time' in df.columns:
            df[df['no_show'] == 0]['booking_lead_time'].hist(alpha=0.6, label='Show', bins=30, color='#2ecc71', ax=ax)
            df[df['no_show'] == 1]['booking_lead_time'].hist(alpha=0.6, label='No-Show', bins=30, color='#e74c3c', ax=ax)
            ax.set_xlabel('Booking Lead Time (Days)')
            ax.set_ylabel('Frequency')
            ax.set_title('Booking Lead Time Distribution')
            ax.legend()
        container.pyplot(fig)
    
    container.markdown("#### 📊 Key Correlations")
    
    if 'ticket_price' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Price distribution by no-show
        axes[0].boxplot([df[df['no_show'] == 0]['ticket_price'].dropna(),
                        df[df['no_show'] == 1]['ticket_price'].dropna()],
                       labels=['Show', 'No-Show'])
        axes[0].set_ylabel('Ticket Price')
        axes[0].set_title('Ticket Price by Show Status')
        
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1], fmt='.2f')
        axes[1].set_title('Feature Correlations')
        
        container.pyplot(fig)


def train_and_evaluate_models(df, container):
    """Train and evaluate ML models."""
    from sklearn.preprocessing import StandardScaler
    
    container.info("🔄 Processing data and training models...")
    
    # Clean data
    df_clean = clean_data(df)
    
    # Engineer features
    df_engineered = engineer_features(df_clean)
    
    # Prepare features
    X, y = prepare_features(df_engineered)
    st.session_state['feature_names'] = list(X.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, stratify=True)
    st.session_state['X_test'] = X_test
    st.session_state['y_test'] = y_test
    
    # Train models
    trainer = ModelTrainer(random_state=42)
    models = trainer.train_all(X_train, y_train, balance_method='weights')
    st.session_state['trainer'] = trainer
    st.session_state['models'] = models
    st.session_state['trained'] = True
    
    # Evaluate models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    comparator = ModelComparator()
    
    results = []
    for name, model_info in models.items():
        model = model_info['model']
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        comparator.add_model(name, y_test.values, y_pred, y_proba)
        
        eval_obj = ModelEvaluator(y_test.values, y_pred, y_proba)
        results.append({
            'Model': name,
            'Accuracy': f"{eval_obj.metrics['accuracy']:.4f}",
            'Precision': f"{eval_obj.metrics['precision']:.4f}",
            'Recall': f"{eval_obj.metrics['recall']:.4f}",
            'F1': f"{eval_obj.metrics['f1_score']:.4f}",
            'ROC-AUC': f"{eval_obj.metrics['roc_auc']:.4f}",
            'PR-AUC': f"{eval_obj.metrics['pr_auc']:.4f}"
        })
    
    results_df = pd.DataFrame(results)
    
    container.success("✅ Models trained successfully!")
    container.markdown("#### 📊 Model Performance Comparison")
    container.dataframe(results_df, use_container_width=True)
    
    # Best model
    best_name, best_eval = comparator.get_best_model('f1_score')
    st.session_state['best_model_name'] = best_name
    container.markdown(f"#### 🏆 Best Model: **{best_name}**")
    
    col1, col2, col3, col4 = container.columns(4)
    col1.metric("Accuracy", f"{best_eval.metrics['accuracy']:.2%}")
    col2.metric("Precision", f"{best_eval.metrics['precision']:.2%}")
    col3.metric("Recall", f"{best_eval.metrics['recall']:.2%}")
    col4.metric("F1-Score", f"{best_eval.metrics['f1_score']:.2%}")
    
    return results_df, trainer, scaler, best_name, best_eval


def predict_no_shows(trainer, scaler, container):
    """Keep local research candidates out of the production prediction path."""
    container.info(
        "Training and evaluation are research-only. Candidate models cannot serve "
        "predictions; use the API only after a model passes the production approval gate."
    )


def generate_recommendations(df, container):
    """Generate business recommendations."""
    container.markdown("#### 📈 Business Insights & Recommendations")
    
    # Calculate key metrics
    noshow_rate = df['no_show'].mean() * 100
    
    col1, col2, col3, col4 = container.columns(4)
    col1.metric("Overall No-Show Rate", f"{noshow_rate:.1f}%")
    
    if 'booking_lead_time' in df.columns:
        avg_lead = df['booking_lead_time'].mean()
        col2.metric("Avg Booking Lead Time", f"{avg_lead:.0f} days")
    
    if 'ticket_price' in df.columns:
        avg_price = df['ticket_price'].mean()
        col3.metric("Avg Ticket Price", f"${avg_price:.0f}")
    
    col4.metric("Potential Impact", f"${avg_price * noshow_rate / 100 * 1000:,.0f}" if 'ticket_price' in df.columns else "N/A")
    
    container.markdown("##### 💡 Key Recommendations")
    
    recommendations = [
        "🔔 **Implement Early Booking Incentives**: Passengers booking less than 7 days before flight have higher no-show rates. Consider offering incentives for early bookings.",
        "📱 **Mobile App Reminders**: Send personalized reminders 48 and 24 hours before the flight to reduce no-shows.",
        "💳 **Flexible Payment Options**: Offer refundable tickets for high-risk passengers at a premium.",
        "🎯 **Targeted Interventions**: Focus resources on passengers with historical no-show patterns.",
        "📊 **Route Optimization**: Identify high no-show routes and adjust overbooking strategies accordingly."
    ]
    
    for rec in recommendations:
        container.markdown(f"""
        <div class="info-box">
            {rec}
        </div>
        """, unsafe_allow_html=True)
    
    # Overbooking simulation section
    container.markdown("##### 🛫 Overbooking Analysis")
    
    col1, col2 = container.columns(2)
    capacity = col1.number_input("Flight Capacity", 100, 500, 180)
    noshow_prob = col2.slider("Avg No-Show Rate", 0.05, 0.30, 0.15, 0.01)
    
    # Simple overbooking calculation
    ticket_price = 350
    empty_seat_cost = 300
    denied_cost = 5000
    compensation = 1500
    
    best_overbook = 0
    best_profit = 0
    
    for overbook in range(0, 21):
        bookings = capacity + overbook
        expected_noshows = bookings * noshow_prob
        expected_shows = bookings - expected_noshows
        
        prob_denied = max(0, (expected_shows - capacity) / bookings) if expected_shows > capacity else 0
        prob_empty = max(0, (capacity - expected_shows) / capacity) if expected_shows < capacity else 0
        
        expected_denied = bookings * prob_denied
        expected_empty = max(0, capacity - expected_shows)
        
        revenue = bookings * ticket_price
        cost = expected_denied * (denied_cost + compensation) + expected_empty * empty_seat_cost
        net = revenue - cost
        
        if net > best_profit:
            best_profit = net
            best_overbook = overbook
    
    container.markdown(f"""
    <div class="success-box">
        <h4>🎯 Optimal Overbooking Recommendation</h4>
        <p><strong>Optimal overbooking level:</strong> +{best_overbook} passengers</p>
        <p><strong>Expected net impact:</strong> ${best_profit:,.0f}</p>
        <p><strong>Configuration:</strong> Capacity: {capacity}, No-Show Rate: {noshow_prob*100:.0f}%</p>
    </div>
    """, unsafe_allow_html=True)


# Main application
def main():
    """Main application function."""
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>✈️ Airline No-Show Prediction System</h1>
        <p>Advanced Machine Learning Analytics for Passenger Behavior Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    menu = ["📤 Upload Data", "📊 Data Analysis", "🤖 Modeling & Prediction", "💡 Business Recommendations"]
    choice = st.sidebar.radio("Go to", menu)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    This system uses machine learning to predict airline passenger no-shows and provides business intelligence for optimization.
    
    **Features:**
    - ML-based predictions
    - SHAP explainability
    - Overbooking simulation
    - Drift monitoring
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")
    if st.session_state['trained']:
        st.sidebar.info("Research candidates evaluated; prediction serving is disabled.")
    if st.session_state['data'] is not None:
        st.sidebar.info(f"📊 {len(st.session_state['data']):,} records loaded")
    
    # Main content area
    if choice == "📤 Upload Data":
        st.markdown('<div class="section-header">📤 Data Upload & Overview</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            **Required Data Format:**
            - CSV file with booking records
            - Key columns: booking_date, flight_date, ticket_price, passenger info
            - Target column: no_show (0/1)
            """)
            
            st.markdown("**Using sample data:**")
            if st.button("Load Sample Data"):
                csv_path = os.path.join(project_root, 'data', 'training_data.csv')
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    st.session_state['data'] = df
                    st.success(f"✅ Loaded {len(df):,} records from training_data.csv")
                else:
                    st.error("Sample data not found. Please upload a CSV file.")
        
        with col2:
            uploaded_file = st.file_uploader("Upload your booking data (CSV)", type=["csv"])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state['data'] = df
                    st.success(f"✅ File uploaded successfully! {len(df):,} records.")
                    
                    st.markdown("#### 📋 Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    st.markdown("#### 📊 Column Information")
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Non-Null': df.count().values,
                        'Dtype': df.dtypes.astype(str).values
                    })
                    st.dataframe(col_info, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
    
    elif choice == "📊 Data Analysis":
        st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
        
        if st.session_state['data'] is not None:
            df = st.session_state['data']
            
            # Clean and prepare data for visualization
            df_clean = clean_data(df)
            df_engineered = engineer_features(df_clean)
            
            run_eda(df_clean, st)
            show_visualizations(df_engineered, st)
        else:
            st.markdown("""
            <div class="warning-box">
                ⚠️ Please upload data first to perform analysis.
            </div>
            """, unsafe_allow_html=True)
    
    elif choice == "🤖 Modeling & Prediction":
        st.markdown('<div class="section-header">🤖 Machine Learning Models</div>', unsafe_allow_html=True)
        
        if st.session_state['data'] is not None:
            df = st.session_state['data']
            
            if not st.session_state['trained']:
                with st.spinner("🔄 Training models..."):
                    results_df, trainer, scaler, best_name, best_eval = train_and_evaluate_models(df, st)
            else:
                st.info("Research candidates are available for evaluation only; they are not approved to serve predictions.")
                st.markdown("#### 🏆 Trained Models")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Models Trained", len(st.session_state['models']))
                col1.metric("Features", len(st.session_state['feature_names']))
            
            # Prediction section
            st.markdown("---")
            st.markdown("### 🔮 Make Predictions")
            
            if st.session_state['trained']:
                predict_no_shows(st.session_state['trainer'], None, st)
            else:
                st.info("Train models first to enable predictions.")
        
        else:
            st.markdown("""
            <div class="warning-box">
                ⚠️ Please upload data first to train models.
            </div>
            """, unsafe_allow_html=True)
    
    elif choice == "💡 Business Recommendations":
        st.markdown('<div class="section-header">💡 Business Intelligence & Recommendations</div>', unsafe_allow_html=True)
        
        if st.session_state['data'] is not None:
            df = st.session_state['data']
            df_clean = clean_data(df)
            generate_recommendations(df_clean, st)
        else:
            st.markdown("""
            <div class="warning-box">
                ⚠️ Please upload data first to generate recommendations.
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 1rem;">
        <p>✈️ Airline No-Show Prediction System | Built with Python, Scikit-learn, and Streamlit</p>
    </div>
    """)


if __name__ == "__main__":
    main()