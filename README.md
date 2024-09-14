
# Remedoc

Remedoc is a revolutionary online all-in-one healthcare service with advanced AI features. This project provides comprehensive medical services through three main offerings: MedSign, DocTalk, and SleepTrip. Each service is powered by state-of-the-art AI technologies to ensure the best user experience.

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- **MedSign**: Utilizes Gemini LLM for medical sign interpretation and diagnostics.
- **DocTalk**: Employs Langchain and Groq API, powered by LLM LLama 3 for doctor-patient interaction.
- **SleepTrip**: Uses Gemini for sleep tracking and recommendations.

## Demo

https://youtu.be/BoVQg5786oQ

Click the link to watch the demo video.

## Technologies Used

### Frontend

- HTML
- Bootstrap
- CSS

### Backend

- Flask

### Services

- **MedSign**: Powered by Gemini LLM
- **DocTalk**: Powered by Langchain, Groq API, and LLM LLama 3
- **SleepTrip**: Powered by Gemini

## Setup and Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/mozito02/Remedoc_main.git
   cd Remedoc_main
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**

   Create a `.env` file in the root directory of the project and add your API keys:

   ```plaintext
   GEMINI_API_KEY=your_gemini_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

5. **Run the Application**

   ```bash
   flask run
   ```

## Environment Variables

The following environment variables are required to run the project:

- `GEMINI_API_KEY`: API key for accessing Gemini services.
- `GROQ_API_KEY`: API key for accessing Groq services.

## Usage

Once the application is running, you can access the Remedoc services through the following endpoints:

- **MedSign**: `/medsign`
- **DocTalk**: `/doctalk`
- **SleepTrip**: `/sleeptrip`

Each service provides a unique set of functionalities tailored to the specific healthcare needs of the user.

## Contributing

We welcome contributions from the community. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your fork.
5. Open a pull request to the main repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
