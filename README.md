# E-commerce Recommendation System

A machine learning-powered e-commerce web application that provides personalized product recommendations using TensorFlow.js. The system analyzes user purchase history and behavior to suggest relevant products through collaborative filtering and neural networks.

## 🚀 Features

- **User Profile Management**: Browse and select user profiles with detailed information
- **Product Catalog**: Interactive product listings with filtering and search capabilities
- **Purchase Tracking**: Real-time tracking of user purchases using sessionStorage
- **Past Purchase History**: Display of user's previous purchases
- **Machine Learning Recommendations**: TensorFlow.js-based recommendation engine using:
  - Collaborative filtering
  - Neural network model training
  - User similarity analysis
- **Model Training Visualization**: Interactive TensorFlow.js Visor for monitoring model performance
- **Web Workers**: Background processing for model training without blocking the UI

## 🛠️ Tech Stack

- **Frontend**: Vanilla JavaScript (ES6+), HTML5, CSS3
- **Machine Learning**: TensorFlow.js
- **Architecture**: MVC Pattern
- **Data Storage**: JSON files, sessionStorage
- **Build Tools**: Webpack Dev Server

## 📁 Project Structure

```
ecommerce-recommendations/
├── index.html                          # Main HTML entry point
├── style.css                           # Global styles
├── src/
│   ├── index.js                        # Application entry point
│   ├── controller/                     # MVC Controllers
│   │   ├── ModelTrainingController.js  # ML model training logic
│   │   ├── ProductController.js        # Product management
│   │   ├── TFVisorController.js        # TensorFlow Visor integration
│   │   ├── UserController.js           # User management
│   │   └── WorkerController.js         # Web Worker coordination
│   ├── service/                        # Business logic layer
│   │   ├── ProductService.js           # Product data operations
│   │   └── UserService.js              # User data operations
│   ├── view/                           # View layer
│   │   ├── ModelTrainingView.js        # ML training UI
│   │   ├── ProductView.js              # Product display UI
│   │   ├── TFVisorView.js              # Visor UI components
│   │   ├── UserView.js                 # User profile UI
│   │   ├── View.js                     # Base view class
│   │   └── templates/                  # HTML templates
│   ├── events/                         # Event management
│   │   ├── constants.js                # Event type constants
│   │   └── events.js                   # Custom event system
│   └── workers/                        # Web Workers
│       └── modelTrainingWorker.js      # Background ML training
├── data/
│   ├── products.json                   # Product catalog data
│   └── users.json                      # User profiles and history
└── package.json                        # Project dependencies
```

## 🔧 Installation & Setup

1. **Clone the repository**:

```bash
git clone https://github.com/RPellicioli/ecommerce-recommendations.git
cd ecommerce-recommendations
```

2. **Install dependencies**:

```bash
npm install
```

3. **Start the development server**:

```bash
npm start
```

4. **Open your browser** and navigate to:

```
http://localhost:3000
```

## 📊 How It Works

1. **Data Collection**: The application tracks user purchases and browsing behavior
2. **Feature Engineering**: User-product interactions are encoded into feature vectors
3. **Model Training**: A neural network is trained using TensorFlow.js to predict user preferences
4. **Recommendations**: The trained model generates personalized product suggestions
5. **Visualization**: TensorFlow.js Visor displays training metrics and model performance

## 🎯 Usage

1. **Select a User**: Choose a user profile from the available list
2. **View Purchase History**: See the user's past purchases
3. **Browse Products**: Explore the product catalog
4. **Make Purchases**: Click "Buy Now" to simulate purchases (tracked in sessionStorage)
5. **Train Model**: Use the model training interface to train the recommendation engine
6. **View Recommendations**: Get personalized product suggestions based on the trained model

## 🧠 Machine Learning Details

The recommendation system uses:

- **Collaborative Filtering**: Analyzes patterns in user-product interactions
- **Neural Network Architecture**: Multi-layer perceptron for learning user preferences
- **Training Strategy**: Background training using Web Workers for non-blocking UI
- **Evaluation Metrics**: Loss and accuracy visualization through TensorFlow.js Visor

## 📦 Dependencies

```json
{
  "devDependencies": {
    "browser-sync": "^3.0.4"
  }
}
```

## 👤 Author

Ricardo Pellicioli - [@RPellicioli](https://github.com/RPellicioli)

## 🙏 Acknowledgments

- TensorFlow.js team for the excellent ML framework
- Inspiration from modern e-commerce recommendation systems
- MVC pattern implementation best practices

---

⭐ Star this repository if you find it helpful!
