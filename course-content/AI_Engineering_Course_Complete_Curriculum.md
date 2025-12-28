# 6-Week AI Engineering Cohort Course
## Complete Curriculum & Content Guide

---

## Course Overview

**Duration:** 6 weeks  
**Time Commitment:** 4-6 hours per week  
**Format:** Live cohort-based learning with recordings  
**Prerequisites:** Basic computer science fundamentals; Python helpful but not required  

**Course Philosophy:** Learn by building. Each week combines theoretical foundations with hands-on projects to create real AI systems.

---

## Week 1: AI Fundamentals & Python for AI

### Learning Objectives
- Understand what AI, Machine Learning, and Deep Learning are
- Set up a complete AI development environment
- Master essential Python libraries for AI (NumPy, Pandas, Matplotlib)
- Build your first simple ML model

### Session 1.1: Introduction to AI (90 minutes)

**Topics:**
1. **What is Artificial Intelligence?** (20 min)
   - Definition and scope of AI
   - Types of AI: Narrow AI vs. General AI
   - Real-world applications across industries
   - Current state and future trends

2. **Machine Learning Fundamentals** (30 min)
   - Supervised vs. Unsupervised vs. Reinforcement Learning
   - The ML workflow: Data → Model → Predictions
   - Key terminology: features, labels, training, testing
   - Common ML algorithms overview

3. **Deep Learning Introduction** (20 min)
   - What makes deep learning "deep"?
   - Neural networks basics
   - When to use deep learning vs. traditional ML
   - Success stories: image recognition, NLP, games

4. **Environment Setup** (20 min)
   - Installing Python and Anaconda
   - Setting up Jupyter Notebooks
   - Installing essential libraries
   - Introduction to Google Colab (cloud alternative)

**Hands-on Activity:**
- Set up your development environment
- Run your first Python notebook
- Load and visualize a simple dataset

---

### Session 1.2: Python for AI (90 minutes)

**Topics:**
1. **Python Refresher** (30 min)
   - Data types, functions, and control flow
   - List comprehensions
   - Working with files
   - Essential Python patterns for AI

2. **NumPy Essentials** (30 min)
   - Arrays and array operations
   - Broadcasting and vectorization
   - Linear algebra operations
   - Why NumPy matters for AI

3. **Pandas for Data Manipulation** (20 min)
   - DataFrames and Series
   - Loading and exploring datasets
   - Data cleaning basics
   - Filtering and aggregating data

4. **Matplotlib for Visualization** (10 min)
   - Creating plots
   - Visualizing distributions
   - Plotting multiple variables

**Hands-on Activity:**
- Work through a data analysis notebook
- Load a real dataset (e.g., Iris or Titanic)
- Perform basic exploratory data analysis
- Create meaningful visualizations

---

### Week 1 Project: Build Your First ML Model

**Project:** House Price Predictor

**Description:** Build a simple linear regression model to predict house prices based on features like size, location, and number of bedrooms.

**Steps:**
1. Load and explore the housing dataset
2. Clean and prepare the data
3. Visualize relationships between features and price
4. Split data into training and testing sets
5. Build a linear regression model using scikit-learn
6. Evaluate model performance
7. Make predictions on new data

**Deliverable:** Jupyter notebook with complete analysis and predictions

**Resources:**
- Dataset: California Housing or Ames Housing dataset
- scikit-learn documentation
- Pandas cheat sheet
- Sample solution notebook (provided after submission)

---

## Week 2: Classical Machine Learning

### Learning Objectives
- Understand key ML algorithms and when to use them
- Master the ML pipeline: preprocessing, training, evaluation
- Learn feature engineering techniques
- Build classification and regression models

### Session 2.1: Core ML Algorithms (90 minutes)

**Topics:**
1. **Linear Models** (25 min)
   - Linear Regression (review from Week 1)
   - Logistic Regression for classification
   - Regularization: Ridge and Lasso
   - When linear models work best

2. **Tree-Based Methods** (25 min)
   - Decision Trees: how they work
   - Random Forests: ensemble learning
   - Gradient Boosting (XGBoost, LightGBM)
   - Handling categorical variables

3. **Other Important Algorithms** (20 min)
   - K-Nearest Neighbors (KNN)
   - Support Vector Machines (SVM)
   - Naive Bayes for text classification

4. **Algorithm Selection Guide** (20 min)
   - Choosing the right algorithm for your problem
   - Trade-offs: accuracy vs. interpretability vs. speed
   - Practical decision framework

**Hands-on Activity:**
- Implement 3-4 different algorithms on the same dataset
- Compare their performance
- Visualize decision boundaries

---

### Session 2.2: The ML Pipeline (90 minutes)

**Topics:**
1. **Data Preprocessing** (30 min)
   - Handling missing values
   - Encoding categorical variables (one-hot, label encoding)
   - Feature scaling and normalization
   - Train-test-validation splits

2. **Feature Engineering** (25 min)
   - Creating new features
   - Polynomial features
   - Interaction terms
   - Domain-specific features
   - Feature selection techniques

3. **Model Evaluation** (25 min)
   - Metrics for classification: accuracy, precision, recall, F1, ROC-AUC
   - Metrics for regression: MSE, RMSE, MAE, R²
   - Confusion matrices
   - Cross-validation

4. **Hyperparameter Tuning** (10 min)
   - Grid search
   - Random search
   - Best practices for avoiding overfitting

**Hands-on Activity:**
- Build a complete ML pipeline using scikit-learn
- Implement cross-validation
- Tune hyperparameters
- Compare multiple models systematically

---

### Week 2 Project: Customer Churn Prediction

**Project:** Build a classifier to predict which customers will leave a service

**Description:** Use a real-world customer dataset to build a classification model that predicts churn. Focus on the complete ML pipeline from data cleaning to model deployment.

**Steps:**
1. Explore and clean the customer dataset
2. Engineer relevant features (customer lifetime value, usage patterns, etc.)
3. Handle imbalanced classes
4. Build and compare multiple classifiers
5. Tune the best performing model
6. Evaluate using appropriate metrics
7. Generate actionable insights for the business

**Deliverable:** 
- Jupyter notebook with complete pipeline
- 1-page report on findings and recommendations

**Resources:**
- Telco Customer Churn dataset
- scikit-learn pipeline documentation
- Class imbalance handling techniques guide

---

## Week 3: Neural Networks & Deep Learning Foundations

### Learning Objectives
- Understand how neural networks learn
- Build neural networks from scratch and with frameworks
- Master PyTorch/TensorFlow basics
- Train deep learning models effectively

### Session 3.1: Neural Networks from Scratch (90 minutes)

**Topics:**
1. **The Neuron: Building Block of Neural Networks** (20 min)
   - Biological inspiration
   - Mathematical model: weights, bias, activation
   - Activation functions: sigmoid, tanh, ReLU
   - From single neuron to layers

2. **Forward Propagation** (20 min)
   - Matrix operations in neural networks
   - Layer-by-layer computation
   - Understanding network architecture
   - Coding a forward pass from scratch

3. **Loss Functions** (15 min)
   - Mean Squared Error for regression
   - Cross-Entropy for classification
   - Why loss functions matter

4. **Backpropagation & Gradient Descent** (25 min)
   - The calculus of learning
   - Chain rule intuition
   - Computing gradients
   - Updating weights
   - Learning rate and convergence

5. **Coding a Neural Network from Scratch** (10 min)
   - Implementing a simple 2-layer network in NumPy
   - Training on XOR problem
   - Understanding what's happening "under the hood"

**Hands-on Activity:**
- Code a simple neural network using only NumPy
- Train it on a toy problem
- Visualize loss decreasing over epochs
- Experiment with learning rates

---

### Session 3.2: Deep Learning Frameworks (90 minutes)

**Topics:**
1. **Introduction to PyTorch** (40 min)
   - Tensors: PyTorch's fundamental data structure
   - Automatic differentiation with autograd
   - Building models with nn.Module
   - Training loops: forward pass, loss, backward pass, optimizer step
   - GPU acceleration basics

2. **Building Your First Deep Network** (30 min)
   - Defining model architecture
   - Choosing optimizers: SGD, Adam, RMSprop
   - Batch processing and DataLoaders
   - Monitoring training with metrics

3. **Common Issues and Solutions** (20 min)
   - Overfitting and underfitting
   - Vanishing and exploding gradients
   - Batch normalization
   - Dropout for regularization
   - Early stopping

**Hands-on Activity:**
- Build a deep neural network in PyTorch
- Train on MNIST digit classification
- Implement dropout and batch normalization
- Plot training and validation curves
- Experiment with different architectures

---

### Week 3 Project: Image Classifier

**Project:** Build a neural network to classify images into multiple categories

**Description:** Create a deep learning model using PyTorch to classify images from the CIFAR-10 dataset (10 categories: airplanes, cars, birds, cats, etc.)

**Steps:**
1. Load and explore CIFAR-10 dataset
2. Preprocess images (normalization, augmentation)
3. Design a neural network architecture
4. Implement training loop with proper validation
5. Add regularization techniques
6. Evaluate model performance
7. Visualize what the network learned
8. Make predictions on new images

**Deliverable:** 
- PyTorch notebook with complete implementation
- Model achieving >70% accuracy on test set
- Visualization of correctly and incorrectly classified images

**Resources:**
- CIFAR-10 dataset
- PyTorch tutorials
- CNN architecture examples
- Data augmentation guide

---

## Week 4: Convolutional Neural Networks (CNNs)

### Learning Objectives
- Understand how CNNs process images
- Build CNN architectures for computer vision
- Apply transfer learning
- Work with pre-trained models

### Session 4.1: CNN Fundamentals (90 minutes)

**Topics:**
1. **Why CNNs for Images?** (15 min)
   - Limitations of fully connected networks for images
   - Spatial structure and local patterns
   - Parameter efficiency

2. **Convolutional Layers** (30 min)
   - Filters and feature maps
   - Stride and padding
   - Understanding what filters learn
   - Visualizing convolutions
   - Multiple channels (RGB images)

3. **Pooling Layers** (15 min)
   - Max pooling and average pooling
   - Reducing spatial dimensions
   - Translation invariance

4. **Classic CNN Architectures** (30 min)
   - LeNet: the pioneer
   - AlexNet: the breakthrough
   - VGG: deeper networks
   - ResNet: skip connections
   - Modern architectures overview

**Hands-on Activity:**
- Build a CNN from scratch
- Visualize learned filters
- Apply filters to images
- Compare CNN vs. fully connected network performance

---

### Session 4.2: Transfer Learning & Advanced Techniques (90 minutes)

**Topics:**
1. **Transfer Learning** (35 min)
   - What is transfer learning?
   - Why it works: learned features are transferable
   - Fine-tuning vs. feature extraction
   - Using pre-trained models (ResNet, VGG, EfficientNet)
   - When to fine-tune vs. train from scratch

2. **Data Augmentation** (20 min)
   - Rotation, flipping, cropping
   - Color jittering
   - Mixup and CutMix
   - Creating a robust augmentation pipeline

3. **Advanced CNN Topics** (20 min)
   - Batch normalization in depth
   - Different pooling strategies
   - Global average pooling
   - Attention mechanisms in CNNs

4. **Practical Tips** (15 min)
   - Debugging CNN training
   - Learning rate scheduling
   - Gradient clipping
   - Mixed precision training

**Hands-on Activity:**
- Load a pre-trained ResNet model
- Fine-tune it on a custom dataset
- Implement comprehensive data augmentation
- Compare performance with and without transfer learning

---

### Week 4 Project: Custom Image Recognition System

**Project:** Build a specialized image classifier using transfer learning

**Description:** Choose a domain (e.g., medical images, satellite imagery, plant species, food items) and build a high-accuracy classifier using transfer learning.

**Steps:**
1. Select and collect/download a specialized dataset
2. Perform exploratory data analysis
3. Set up data augmentation pipeline
4. Load a pre-trained model (ResNet50 or EfficientNet)
5. Freeze/unfreeze layers strategically
6. Fine-tune on your custom dataset
7. Evaluate and analyze errors
8. Deploy model for inference
9. Create a simple demo interface

**Deliverable:** 
- Complete training notebook
- Trained model achieving >85% accuracy
- Analysis of failure cases
- Simple inference script or Gradio demo

**Resources:**
- Pre-trained model zoo
- Domain-specific dataset suggestions
- Transfer learning best practices guide
- Gradio documentation for demos

---

## Week 5: Natural Language Processing (NLP) & Transformers

### Learning Objectives
- Understand how to process text data
- Learn the Transformer architecture
- Work with pre-trained language models
- Build NLP applications

### Session 5.1: NLP Fundamentals & Text Processing (90 minutes)

**Topics:**
1. **Introduction to NLP** (20 min)
   - What makes text different from images?
   - Common NLP tasks: classification, NER, QA, translation
   - Challenges in NLP: ambiguity, context, languages

2. **Text Preprocessing** (25 min)
   - Tokenization
   - Lowercasing and punctuation handling
   - Stop words removal
   - Stemming and lemmatization
   - Building vocabularies

3. **Text Representation** (25 min)
   - Bag of Words (BoW)
   - TF-IDF
   - Word embeddings: Word2Vec, GloVe
   - Understanding semantic similarity
   - Visualizing embeddings

4. **RNNs and LSTMs** (20 min)
   - Sequential data processing
   - Hidden states and memory
   - Why LSTMs solve vanishing gradients
   - Limitations of RNNs

**Hands-on Activity:**
- Preprocess a text dataset
- Create word embeddings
- Visualize word relationships
- Build a simple RNN for sentiment analysis

---

### Session 5.2: Transformers & Language Models (90 minutes)

**Topics:**
1. **The Transformer Revolution** (30 min)
   - Attention mechanism: "Attention is All You Need"
   - Self-attention: how it works
   - Multi-head attention
   - Positional encoding
   - Encoder-decoder architecture

2. **Pre-trained Language Models** (25 min)
   - BERT: bidirectional understanding
   - GPT: generative pre-training
   - T5, BART, and others
   - How these models are trained
   - The power of pre-training

3. **Using Hugging Face Transformers** (25 min)
   - The transformers library overview
   - Loading pre-trained models
   - Tokenizers and pipelines
   - Fine-tuning for specific tasks
   - Inference and deployment

4. **Prompt Engineering Basics** (10 min)
   - Crafting effective prompts
   - Few-shot learning
   - Chain-of-thought prompting

**Hands-on Activity:**
- Use Hugging Face pipelines for various NLP tasks
- Fine-tune BERT on sentiment analysis
- Experiment with GPT for text generation
- Compare different model architectures

---

### Week 5 Project: Text Analysis Application

**Project:** Build a multi-task NLP system

**Description:** Create an NLP application that performs multiple tasks: sentiment analysis, named entity recognition, and text summarization on customer reviews or news articles.

**Steps:**
1. Collect or download a text dataset
2. Perform text preprocessing and EDA
3. Implement three NLP tasks:
   - Sentiment classification (fine-tune BERT)
   - Named entity recognition
   - Text summarization (using T5 or BART)
4. Create a unified pipeline
5. Evaluate each component
6. Build a demo interface
7. Analyze and visualize results

**Deliverable:** 
- Three separate notebooks/scripts for each task
- Integrated pipeline
- Interactive demo (Gradio or Streamlit)
- Performance report with examples

**Resources:**
- Hugging Face model hub
- Dataset suggestions (IMDB, Amazon reviews, news articles)
- Fine-tuning guide
- Streamlit/Gradio tutorials

---

## Week 6: LLMs, Agents & Real-World Deployment

### Learning Objectives
- Understand Large Language Models (LLMs)
- Build AI agents that can use tools
- Deploy models to production
- Learn MLOps basics

### Session 6.1: LLMs & AI Agents (90 minutes)

**Topics:**
1. **Large Language Models** (25 min)
   - What makes LLMs "large"?
   - Training LLMs: pre-training and fine-tuning
   - In-context learning
   - Capabilities and limitations
   - GPT, Claude, Llama, and other models

2. **Working with LLM APIs** (20 min)
   - OpenAI API, Anthropic API basics
   - Crafting system prompts
   - Managing context windows
   - Streaming responses
   - Cost optimization

3. **Building AI Agents** (30 min)
   - What is an AI agent?
   - ReAct pattern: Reasoning + Acting
   - Tool use and function calling
   - Memory and state management
   - Agent frameworks: LangChain, LlamaIndex

4. **RAG: Retrieval-Augmented Generation** (15 min)
   - Why RAG matters
   - Vector databases
   - Embedding text for retrieval
   - Combining retrieval with generation

**Hands-on Activity:**
- Call LLM APIs with different prompts
- Build a simple agent that uses tools (calculator, search)
- Implement basic RAG system
- Create a chatbot with memory

---

### Session 6.2: Model Deployment & MLOps (90 minutes)

**Topics:**
1. **Model Serving** (25 min)
   - Saving and loading models
   - ONNX for cross-framework compatibility
   - REST APIs with FastAPI
   - Batch vs. real-time inference
   - Handling requests efficiently

2. **Deployment Options** (20 min)
   - Cloud platforms: AWS, GCP, Azure
   - Hugging Face Spaces
   - Docker containers
   - Serverless deployment
   - Edge deployment considerations

3. **MLOps Fundamentals** (25 min)
   - Version control for models and data
   - Experiment tracking with MLflow/Weights & Biases
   - Model monitoring
   - A/B testing
   - Continuous training

4. **Production Best Practices** (20 min)
   - Input validation
   - Error handling
   - Logging and monitoring
   - Security considerations
   - Scaling considerations
   - Cost optimization

**Hands-on Activity:**
- Create a FastAPI endpoint for a model
- Containerize a model with Docker
- Deploy to Hugging Face Spaces
- Set up basic monitoring

---

### Week 6 Final Project: End-to-End AI Application

**Project:** Build and Deploy a Complete AI System

**Description:** Create a full-stack AI application that combines multiple concepts from the course. Choose one of the following or propose your own:

**Option 1: AI-Powered Customer Support Agent**
- RAG system with company documentation
- Sentiment analysis of customer messages
- Response generation with LLM
- Web interface for interaction

**Option 2: Content Moderation System**
- Image classification for inappropriate content
- Text toxicity detection
- Multi-modal analysis
- Dashboard for moderators

**Option 3: Personal AI Assistant**
- Task classification
- Entity extraction
- Tool use (calendar, reminders, search)
- Conversational interface

**Option 4: Document Intelligence System**
- OCR and document parsing
- Classification and information extraction
- Q&A over documents
- Summary generation

**Requirements:**
1. Use at least 2 different AI models/techniques
2. Create a user interface (web or CLI)
3. Deploy the application
4. Document your system architecture
5. Include error handling and monitoring
6. Prepare a demo video

**Deliverable:** 
- Complete codebase (GitHub repository)
- Deployed application (live link)
- Documentation and README
- 5-minute demo video
- Presentation deck

**Resources:**
- Full course materials and past projects
- Deployment guides for various platforms
- UI framework tutorials (Streamlit/Gradio)
- Example architectures

---

## Course Completion

### Final Week Activities

1. **Project Presentations** (120 minutes)
   - Each student presents their final project (5-7 minutes)
   - Q&A and peer feedback
   - Best project showcase

2. **Course Review & Next Steps** (60 minutes)
   - Key concepts recap
   - Resource recommendations for continued learning
   - Career paths in AI
   - Building your AI portfolio
   - Staying current with AI developments

3. **Graduation & Community**
   - Course completion certificates
   - Join alumni community
   - Access to continued resources
   - Optional office hours

---

## Additional Resources

### Weekly Resources
- Reading materials and papers
- Code templates and examples
- Dataset links
- Tool documentation
- Supplementary videos

### Community Support
- Discord/Slack channel for peer discussion
- Weekly office hours
- Peer code review sessions
- Project feedback forums

### Assessment
- Weekly mini-quizzes (optional, for self-assessment)
- Project evaluations with detailed feedback
- Peer project reviews
- Final project presentation

---

## Success Tips

1. **Complete the hands-on activities** - Don't just watch, code along
2. **Start projects early** - Give yourself time to experiment
3. **Ask questions** - Use live sessions to get clarification
4. **Connect with peers** - Learn from others' approaches
5. **Build a portfolio** - Document your projects publicly
6. **Review recordings** - Reinforce concepts by rewatching
7. **Experiment** - Try variations beyond assignments
8. **Stay curious** - Explore topics that interest you most

---

## Prerequisites Refresher

If you need to brush up on prerequisites, here are some resources:

**Python Basics:**
- Variables, data types, loops
- Functions and modules
- List/dictionary operations
- File I/O

**Math Concepts (helpful but not required):**
- Basic algebra
- Matrix operations
- Derivatives (intuition, not calculus)
- Probability basics

**Computer Science:**
- Basic programming logic
- Understanding of algorithms (high-level)
- File systems and paths

---

## Instructor Office Hours

- Weekly office hours: [Day/Time TBD]
- 1-on-1 project consultations available
- Code review sessions
- Career guidance discussions

---

## Course Evolution

This curriculum is a living document. Based on cohort feedback and emerging AI developments, content may be adjusted to ensure you're learning the most relevant and practical skills.

---

**Ready to start your AI engineering journey? Let's build something amazing together! 🚀**
