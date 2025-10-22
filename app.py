"""
Movie Genre Classification App
A Streamlit application for predicting movie genres from descriptions
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PAGE_CONFIG = {
    "page_title": "Movie Genre Classifier",
    "page_icon": "🎬",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

GENRES = ["drama", "documentary", "comedy", "short", "horror", "thriller", "action", "western"]

GENRE_STATS = {
    'Genre': ['Drama', 'Documentary', 'Comedy', 'Short', 'Horror', 'Thriller', 'Action', 'Western'],
    'Count': [13613, 13096, 7447, 5073, 2204, 1591, 1315, 1032]
}

# ============================================================================
# INITIALIZATION
# ============================================================================

@st.cache_resource
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('omw-1.4', quiet=True)

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

class TextPreprocessor:
    """Enhanced text preprocessing class"""
    
    def __init__(self):
        download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def process(self, text):
        """Process text with cleaning, tokenization, and lemmatization"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Keep only alphanumeric characters and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
        
        # Tokenization
        try:
            words = word_tokenize(text)
        except:
            words = text.split()
        
        # Remove stopwords and short words
        words = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        # Lemmatization
        words = [self.lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained model with error handling"""
    model_path = Path('models/best_genre_classifier_LinearSVC.pkl')
    
    if not model_path.exists():
        st.error("⚠️ Model file 'best_genre_classifier_LinearSVC.pkl' not found. Please ensure the model is trained and saved.")
        st.info("Train the model using the notebook: Datapreocessing&EDA&modeling.ipynb")
        return None
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_prediction_confidence(model, processed_text):
    """Get prediction confidence scores, handling models with/without predict_proba"""
    try:
        # Try to get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba([processed_text])[0]
        else:
            # For models without predict_proba, use decision function or create dummy confidences
            if hasattr(model, 'decision_function'):
                decision_scores = model.decision_function([processed_text])[0]
                # Convert decision scores to pseudo-probabilities using softmax
                exp_scores = np.exp(decision_scores - np.max(decision_scores))
                probabilities = exp_scores / np.sum(exp_scores)
            else:
                # Fallback: equal confidence for all classes
                probabilities = np.ones(len(model.classes_)) / len(model.classes_)
        return probabilities
    except Exception as e:
        st.warning(f"Could not calculate confidence scores: {str(e)}")
        # Return equal probabilities as fallback
        return np.ones(len(model.classes_)) / len(model.classes_)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confidence_chart(confidence_df):
    """Create an interactive confidence chart using Plotly"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=confidence_df['Genre'],
        x=confidence_df['Confidence'],
        orientation='h',
        marker=dict(
            color=confidence_df['Confidence'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Confidence")
        ),
        text=[f"{val:.1%}" for val in confidence_df['Confidence']],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Prediction Confidence by Genre',
        xaxis_title='Confidence Score',
        yaxis_title='Genre',
        height=400,
        showlegend=False,
        xaxis=dict(tickformat='.0%')
    )
    
    return fig

def plot_genre_distribution():
    """Plot genre distribution from training data"""
    df = pd.DataFrame(GENRE_STATS)
    
    fig = px.bar(
        df, 
        x='Count', 
        y='Genre',
        orientation='h',
        color='Count',
        color_continuous_scale='Viridis',
        title='Distribution of Movie Genres in Training Data'
    )
    
    fig.update_layout(
        xaxis_title='Number of Movies',
        yaxis_title='Genre',
        height=450,
        showlegend=False
    )
    
    return fig

# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def page_genre_prediction():
    """Genre prediction page"""
    st.header("🎯 Predict Movie Genre")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📝 Single Prediction", "📄 Batch Prediction"])
    
    # Tab 1: Single prediction
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            movie_title = st.text_input(
                "Movie Title (optional)", 
                placeholder="e.g., The Shawshank Redemption"
            )
            
            movie_description = st.text_area(
                "Movie Description*",
                placeholder="Enter a detailed movie description here...\n\nExample: 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.'",
                height=200
            )
        
        with col2:
            st.markdown("### 💡 Tips")
            st.info(
                "• Provide detailed descriptions\n"
                "• Include plot elements\n"
                "• Mention themes or tone\n"
                "• More context = better accuracy"
            )
        
        if st.button("🔮 Predict Genre", type="primary", use_container_width=True):
            if not movie_description.strip():
                st.warning("⚠️ Please enter a movie description.")
                return
            
            # Load model
            model = load_model()
            if model is None:
                return
            
            # Preprocess text
            with st.spinner("Processing text..."):
                preprocessor = TextPreprocessor()
                processed_text = preprocessor.process(movie_description)
            
            # Make prediction
            with st.spinner("Making prediction..."):
                prediction = model.predict([processed_text])[0]
                probabilities = get_prediction_confidence(model, processed_text)
            
            # Display results
            st.success(f"### 🎬 Predicted Genre: **{prediction.upper()}**")
            
            # Confidence visualization
            st.subheader("📊 Prediction Confidence")
            
            confidence_df = pd.DataFrame({
                'Genre': [g.capitalize() for g in model.classes_],
                'Confidence': probabilities
            }).sort_values('Confidence', ascending=False)
            
            # Plot
            fig = plot_confidence_chart(confidence_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            st.dataframe(
                confidence_df.style.format({'Confidence': '{:.2%}'}).background_gradient(
                    subset=['Confidence'], 
                    cmap='Greens'
                ),
                use_container_width=True,
                hide_index=True
            )
    
    # Tab 2: Batch prediction
    with tab2:
        st.markdown("Upload a CSV file with movie descriptions for batch prediction.")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV must contain a 'Description' column"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! ({len(df)} rows)")
                
                # Preview
                with st.expander("📋 Preview uploaded data", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Validate columns
                if 'Description' not in df.columns:
                    st.error("❌ CSV file must contain a 'Description' column.")
                    st.info(f"Available columns: {', '.join(df.columns)}")
                    return
                
                # Predict button
                if st.button("🚀 Predict Genres for All Movies", type="primary"):
                    model = load_model()
                    if model is None:
                        return
                    
                    # Process
                    with st.spinner(f"Processing {len(df)} movies..."):
                        preprocessor = TextPreprocessor()
                        df['Processed_Description'] = df['Description'].apply(preprocessor.process)
                        
                        # Predict
                        predictions = model.predict(df['Processed_Description'])
                        
                        # Get confidence scores
                        confidence_scores = []
                        for text in df['Processed_Description']:
                            probs = get_prediction_confidence(model, text)
                            confidence_scores.append(np.max(probs))
                        
                        # Add results
                        df['Predicted_Genre'] = predictions
                        df['Confidence'] = confidence_scores
                    
                    st.success("✅ Predictions completed!")
                    
                    # Display results
                    result_cols = ['Description', 'Predicted_Genre', 'Confidence']
                    if 'Title' in df.columns:
                        result_cols.insert(0, 'Title')
                    
                    st.dataframe(
                        df[result_cols].style.format({'Confidence': '{:.2%}'}).background_gradient(
                            subset=['Confidence'],
                            cmap='Greens'
                        ),
                        use_container_width=True
                    )
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name="movie_genre_predictions.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
                    # Statistics
                    st.subheader("📈 Batch Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Movies", len(df))
                    with col2:
                        st.metric("Average Confidence", f"{df['Confidence'].mean():.1%}")
                    with col3:
                        most_common = df['Predicted_Genre'].mode()[0]
                        st.metric("Most Common Genre", most_common.capitalize())
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

def page_model_info():
    """Model information page"""
    st.header("ℹ️ Model Information")
    
    # Model architecture
    st.subheader("🏗️ Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Pipeline Components
        
        1. **Text Preprocessing**
           - Lowercase conversion
           - URL and HTML removal
           - Tokenization
           - Stopword removal
           - Lemmatization
        
        2. **Feature Extraction**
           - TF-IDF Vectorization
           - N-gram range: (1, 2)
           - Max features: 10,000
        
        3. **Classification Algorithm**
           - Linear Support Vector Classification (LinearSVC)
           - Optimized hyperparameters
           - Multi-class classification
        """)
    
    with col2:
        st.markdown("""
        #### Training Details
        
        - **Training Dataset**: 54,214 movies
        - **Test Dataset**: 54,200 movies
        - **Number of Genres**: 8
        - **Evaluation Metric**: Accuracy, F1-score
        
        #### Performance Metrics
        
        - **Overall Accuracy**: ~85%
        - **Macro F1-Score**: ~82%
        - **Training Time**: ~3 minutes
        
        *Note: Performance varies by genre*
        """)
    
    # Model limitations note
    st.info("""
    **Note**: The current model (LinearSVC) provides confidence scores using approximation methods. 
    For more accurate probability estimates, consider using Logistic Regression or Random Forest in future versions.
    """)
    
    # Supported genres
    st.subheader("🎭 Supported Genres")
    
    genre_cols = st.columns(4)
    for idx, genre in enumerate(GENRES):
        with genre_cols[idx % 4]:
            st.markdown(f"• **{genre.capitalize()}**")
    
    # Model file info
    st.subheader("📦 Model Files")
    
    model_exists = Path('models/best_genre_classifier_LinearSVC.pkl').exists()
    
    st.markdown(f"""
    - **best_genre_classifier_LinearSVC.pkl**: {'✅ Found' if model_exists else '❌ Not found'}
    """)

def page_data_exploration():
    """Data exploration page"""
    st.header("📊 Data Exploration")
    
    # Dataset overview
    st.subheader("📖 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Samples", "54,214")
    with col2:
        st.metric("Test Samples", "54,200")
    with col3:
        st.metric("Total Genres", "8")
    with col4:
        st.metric("Vocabulary Size", "~50K")
    
    # Genre distribution
    st.subheader("📈 Genre Distribution")
    fig = plot_genre_distribution()
    st.plotly_chart(fig, use_container_width=True)
    
    # Text statistics
    st.subheader("📝 Text Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Description Length
        - **Average**: 404 characters
        - **Median**: 380 characters
        - **Range**: 50 - 2,000 characters
        """)
    
    with col2:
        st.markdown("""
        #### Word Count
        - **Average**: 57 words
        - **Median**: 52 words
        - **Range**: 10 - 300 words
        """)
    
    # Dataset source
    st.subheader("🔗 Dataset Information")
    st.info("""
    **Source**: IMDB Genre Classification Dataset
    
    This dataset contains movie descriptions from IMDB across multiple genres.
    The data has been preprocessed and filtered to include only the 8 most common genres
    for better model performance and clearer classification boundaries.
    """)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    
    # Page configuration
    st.set_page_config(**PAGE_CONFIG)
    
    # Custom CSS
    st.markdown("""
    <style>
        .main > div {
            padding-top: 2rem;
        }
        .stButton > button {
            width: 100%;
        }
        h1 {
            color: #FF4B4B;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("🎬 Movie Genre Classifier")
    st.markdown("Predict movie genres from descriptions using machine learning")
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/movie.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select a page:",
            ["🎯 Genre Prediction", "ℹ️ Model Information", "📊 Data Exploration"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Instructions
        with st.expander("📖 How to Use", expanded=False):
            st.markdown("""
            1. **Genre Prediction**: Enter a description or upload CSV
            2. **Model Information**: Learn about the model
            3. **Data Exploration**: Explore training data stats
            """)
        
        # About
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
            This app uses machine learning to classify movie genres
            based on their descriptions.
            
            **Tech Stack:**
            - Streamlit
            - scikit-learn
            - NLTK
            - Plotly
            """)
    
    # Route to selected page
    if page == "🎯 Genre Prediction":
        page_genre_prediction()
    elif page == "ℹ️ Model Information":
        page_model_info()
    elif page == "📊 Data Exploration":
        page_data_exploration()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with ❤️ using Streamlit | Movie Genre Classification App"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()