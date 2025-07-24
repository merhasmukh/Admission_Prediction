import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, date
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from prediction import AdmissionPredictor
except ImportError:
    st.error("Please run 'python train_models.py' first to train the models.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="University Admission Prediction Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    """Load the admission predictor (cached)"""
    try:
        predictor = AdmissionPredictor()
        return predictor
    except Exception as e:
        st.error(f"Error loading predictor: {str(e)}")
        return None

def create_gauge_chart(value, title, max_value=1.0):
    """Create a gauge chart for probabilities"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, max_value], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def display_prediction_results(results):
    """Display prediction results in a formatted way"""
    st.subheader("🔮 Prediction Results")
    
    # Overall metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_class = f"risk-{results['risk_assessment'].lower()}"
        st.markdown(f"**Overall Risk:** <span class='{risk_class}'>{results['risk_assessment']}</span>", 
                   unsafe_allow_html=True)
    
    with col2:
        st.metric("Journey Success Probability", f"{results['journey_probability']:.1%}")
    
    with col3:
        total_predictions = len([p for p in results['predictions'].values() if 'error' not in p])
        st.metric("Predictions Made", total_predictions)
    
    # Detailed predictions
    st.subheader("📊 Stage-wise Predictions")
    
    predictions = results['predictions']
    
    # Create columns for each prediction stage
    stages = ['admission_confirmation', 'joining', 'dropout']
    stage_names = ['Admission Confirmation', 'Joining Decision', 'Dropout Risk']
    
    cols = st.columns(len(stages))
    
    for i, (stage, stage_name) in enumerate(zip(stages, stage_names)):
        with cols[i]:
            if stage in predictions and 'error' not in predictions[stage]:
                pred = predictions[stage]
                
                # Create gauge chart
                confidence = pred.get('confidence', 0.5)
                fig = create_gauge_chart(confidence, stage_name)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display prediction details
                st.write(f"**Prediction:** {pred['prediction_label']}")
                if confidence:
                    st.write(f"**Confidence:** {confidence:.1%}")
                
                # Model performance
                if 'model_performance' in pred and pred['model_performance']:
                    perf = pred['model_performance']
                    st.write(f"**Model F1-Score:** {perf.get('f1', 'N/A'):.3f}")
            else:
                st.info(f"{stage_name} prediction not available")

def create_batch_analysis_charts(batch_results):
    """Create charts for batch analysis"""
    # Extract data for visualization
    data = []
    for result in batch_results:
        student_info = result['student_info']
        predictions = result['predictions']
        
        row = {
            'Student_ID': result.get('student_id', 'Unknown'),
            'Risk_Assessment': result['risk_assessment'],
            'Journey_Probability': result['journey_probability'],
            'Marks_12th': student_info.get('Marks_12th_Percent', 0),
            'Entrance_Score': student_info.get('Entrance_Exam_Score', 0),
            'Distance': student_info.get('Distance_From_Home_KM', 0),
            'Gender': student_info.get('Gender', 'Unknown'),
            'Category': student_info.get('Category', 'Unknown')
        }
        
        # Add prediction results
        for stage in ['admission_confirmation', 'joining', 'dropout']:
            if stage in predictions and 'error' not in predictions[stage]:
                row[f'{stage}_prediction'] = predictions[stage]['prediction']
                row[f'{stage}_confidence'] = predictions[stage].get('confidence', 0.5)
            else:
                row[f'{stage}_prediction'] = None
                row[f'{stage}_confidence'] = None
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Create visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Risk Distribution', 'Journey Probability vs Academic Performance', 
                       'Predictions by Gender', 'Distance vs Risk'),
        specs=[[{"type": "pie"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Risk distribution pie chart
    risk_counts = df['Risk_Assessment'].value_counts()
    fig.add_trace(
        go.Pie(labels=risk_counts.index, values=risk_counts.values, name="Risk"),
        row=1, col=1
    )
    
    # Journey probability vs academic performance
    fig.add_trace(
        go.Scatter(
            x=df['Marks_12th'], 
            y=df['Journey_Probability'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['Entrance_Score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Entrance Score")
            ),
            text=df['Student_ID'],
            name="Students"
        ),
        row=1, col=2
    )
    
    # Predictions by gender
    gender_risk = df.groupby(['Gender', 'Risk_Assessment']).size().unstack(fill_value=0)
    for risk_level in gender_risk.columns:
        fig.add_trace(
            go.Bar(x=gender_risk.index, y=gender_risk[risk_level], name=risk_level),
            row=2, col=1
        )
    
    # Distance vs risk
    risk_colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    for risk in risk_colors:
        risk_data = df[df['Risk_Assessment'] == risk]
        fig.add_trace(
            go.Scatter(
                x=risk_data['Distance'],
                y=risk_data['Journey_Probability'],
                mode='markers',
                marker=dict(color=risk_colors[risk], size=8),
                name=f'{risk} Risk',
                text=risk_data['Student_ID']
            ),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=True)
    return fig, df

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🎓 University Admission Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load predictor
    predictor = load_predictor()
    if predictor is None:
        st.error("Failed to load prediction models. Please train the models first.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Single Student Prediction", "Batch Analysis", "Model Information"])
    
    if page == "Single Student Prediction":
        st.header("👤 Single Student Prediction")
        
        # Input form
        with st.form("student_form"):
            st.subheader("Student Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                app_id = st.text_input("Application ID", value="PRED001")
                gender = st.selectbox("Gender", ["Male", "Female"])
                marks_12th = st.number_input("12th Grade Marks (%)", min_value=0.0, max_value=100.0, value=75.0)
                entrance_score = st.number_input("Entrance Exam Score", min_value=0.0, max_value=100.0, value=60.0)
                category = st.selectbox("Category", ["General", "OBC", "SC", "ST"])
            
            with col2:
                admission_round = st.selectbox("Admission Round", [1, 2, 3])
                distance = st.number_input("Distance from Home (KM)", min_value=0, max_value=1000, value=100)
                family_income = st.selectbox("Family Income", ["Low", "Middle", "High"])
                first_preference = st.selectbox("First Preference", ["Yes", "No"])
                form_date = st.date_input("Form Submission Date", value=date(2024, 6, 15))
            
            submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Prepare student data
            student_data = {
                'Application_ID': app_id,
                'Gender': gender,
                'Marks_12th_Percent': marks_12th,
                'Entrance_Exam_Score': entrance_score,
                'Form_Submission_Date': form_date.strftime('%Y-%m-%d'),
                'Category': category,
                'Admission_Round': admission_round,
                'Distance_From_Home_KM': distance,
                'Family_Income': family_income,
                'First_Preference': 1 if first_preference == "Yes" else 0
            }
            
            # Make prediction
            with st.spinner("Making predictions..."):
                results = predictor.predict_complete_journey(student_data)
            
            # Display results
            display_prediction_results(results)
            
            # Risk profile
            st.subheader("⚠️ Risk Profile")
            risk_profile = predictor.create_risk_profile(student_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Risk Factors:**")
                if risk_profile['risk_factors']:
                    for factor in risk_profile['risk_factors']:
                        st.write(f"• {factor}")
                else:
                    st.write("No significant risk factors identified")
            
            with col2:
                st.write("**Recommendations:**")
                if risk_profile['recommendations']:
                    for rec in risk_profile['recommendations']:
                        st.write(f"• {rec}")
                else:
                    st.write("No specific recommendations")
    
    elif page == "Batch Analysis":
        st.header("📊 Batch Analysis")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file with student data", type=['csv'])
        
        if uploaded_file is not None:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} students")
            
            # Show sample data
            with st.expander("View sample data"):
                st.dataframe(df.head())
            
            if st.button("Run Batch Prediction"):
                with st.spinner("Processing batch predictions..."):
                    batch_results = predictor.predict_batch(df)
                
                # Create visualizations
                fig, results_df = create_batch_analysis_charts(batch_results)
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.subheader("📈 Summary Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    high_risk = len([r for r in batch_results if r['risk_assessment'] == 'High'])
                    st.metric("High Risk Students", high_risk)
                
                with col2:
                    avg_journey_prob = np.mean([r['journey_probability'] for r in batch_results])
                    st.metric("Avg Journey Success", f"{avg_journey_prob:.1%}")
                
                with col3:
                    confirmed_pred = len([r for r in batch_results 
                                        if 'admission_confirmation' in r['predictions'] 
                                        and r['predictions']['admission_confirmation'].get('prediction') == 1])
                    st.metric("Predicted Confirmations", confirmed_pred)
                
                with col4:
                    dropout_pred = len([r for r in batch_results 
                                      if 'dropout' in r['predictions'] 
                                      and r['predictions']['dropout'].get('prediction') == 1])
                    st.metric("Predicted Dropouts", dropout_pred)
                
                # Download results
                st.subheader("💾 Download Results")
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Prediction Results",
                    data=csv,
                    file_name=f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("Please upload a CSV file to perform batch analysis")
            
            # Show expected format
            st.subheader("Expected CSV Format")
            sample_data = {
                'Application_ID': ['APP001', 'APP002'],
                'Gender': ['Male', 'Female'],
                'Marks_12th_Percent': [78.5, 82.0],
                'Entrance_Exam_Score': [65.0, 70.0],
                'Form_Submission_Date': ['2024-06-15', '2024-06-20'],
                'Category': ['General', 'OBC'],
                'Admission_Round': [1, 1],
                'Distance_From_Home_KM': [150, 200],
                'Family_Income': ['Middle', 'High'],
                'First_Preference': [1, 1]
            }
            st.dataframe(pd.DataFrame(sample_data))
    
    elif page == "Model Information":
        st.header("🤖 Model Information")
        
        # Get model info
        model_info = predictor.get_model_info()
        
        # Display model status
        st.subheader("Model Status")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Loaded Models:**")
            for model in model_info['models_loaded']:
                st.write(f"✅ {model.replace('_', ' ').title()}")
        
        with col2:
            preprocessor_status = "✅ Loaded" if model_info['preprocessor_loaded'] else "❌ Not Loaded"
            st.write(f"**Preprocessor:** {preprocessor_status}")
        
        # Model performance
        if model_info['model_performance']:
            st.subheader("📊 Model Performance")
            
            perf_data = []
            for model_name, metrics in model_info['model_performance'].items():
                perf_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Algorithm': metrics.get('model_name', 'Unknown'),
                    'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
                    'Precision': f"{metrics.get('precision', 0):.3f}",
                    'Recall': f"{metrics.get('recall', 0):.3f}",
                    'F1-Score': f"{metrics.get('f1', 0):.3f}"
                })
            
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True)
        
        # System information
        st.subheader("🔧 System Information")
        st.write(f"**Dashboard Version:** 1.0.0")
        st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Instructions
        st.subheader("📖 Instructions")
        st.markdown("""
        **How to use this dashboard:**
        
        1. **Single Student Prediction**: Enter individual student details to get predictions
        2. **Batch Analysis**: Upload a CSV file to analyze multiple students at once
        3. **Model Information**: View model performance and system status
        
        **Tips:**
        - Ensure all required fields are filled for accurate predictions
        - Higher academic scores generally lead to better outcomes
        - Distance from home and family income are important factors
        - First preference and early admission rounds improve success rates
        """)

if __name__ == "__main__":
    main()
