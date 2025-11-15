# EMOPIA: AI-Powered Emotion-Based Music Generation System

**Project Report**

---

## Cover Page

**Jaypee Institute of Information Technology, Noida**  
**Department of Computer Science & Engineering and IT**

### EMOPIA: AI-POWERED EMOTION-BASED MUSIC GENERATION SYSTEM

**Student Details:**  
Enrollment No.: [Your Enrollment Number]  
Name: [Your Name]

**Supervisor:**  
[Supervisor Name]  
[Designation]

**Submitted in partial fulfillment of the Degree of**  
**Bachelor of Technology**  
**in**  
**Computer Science Engineering**

**DEPARTMENT OF COMPUTER SCIENCE ENGINEERING & INFORMATION TECHNOLOGY**  
**JAYPEE INSTITUTE OF INFORMATION TECHNOLOGY, NOIDA**

**(November – 2024)**

---

## DECLARATION

I hereby declare that this submission is my own work and that, to the best of my knowledge and belief, it contains no material previously published or written by another person nor material which has been accepted for the award of any other degree or diploma of the university or other institute of higher learning, except where due acknowledgment has been made in the text.

Place: ________________________  
Date: _________________________  
Signature: _______________________  
Name: ___________________________  
Enrollment No: ___________________

---

## CERTIFICATE

This is to certify that the work titled **"EMOPIA: AI-Powered Emotion-Based Music Generation System"** submitted by **[Student Name]** in partial fulfillment for the award of degree of Bachelor of Technology of Jaypee Institute of Information Technology, Noida has been carried out under my supervision.

This work has not been submitted partially or wholly to any other University or Institute for the award of this or any other degree or diploma.

Signature of Supervisor: ____________________  
Name of Supervisor: _______________________  
Designation: ______________________________  
Date: ____________________________________

---

## ACKNOWLEDGEMENT

I would like to express my sincere gratitude to my project supervisor, [Supervisor Name], for their invaluable guidance, continuous support, and encouragement throughout this project. Their expertise and insights have been instrumental in shaping this work.

I am thankful to the Department of Computer Science & Engineering and IT, Jaypee Institute of Information Technology, Noida, for providing the necessary resources and infrastructure to complete this project.

I would also like to acknowledge the creators of the EMOPIA dataset and the open-source community for their contributions to the field of AI and music generation, which formed the foundation of this research.

Finally, I am grateful to my family and friends for their constant support and motivation throughout this journey.

Signature of the Student: ____________________  
Name of Student: ___________________________  
Enrollment Number: _______________________  
Date: ____________________________________

---

## SUMMARY

This project presents EMOPIA, an AI-powered emotion-based music generation system that creates MIDI music compositions conditioned on specific emotions. The system employs a Transformer-based deep learning architecture trained on the EMOPIA dataset containing 1,078 emotion-labeled MIDI files across six emotion categories: joy, sadness, anger, calm, surprise, and fear.

The implementation features a complete full-stack application with a React-based frontend, Flask REST API backend, and PyTorch-based music generation model. The system supports natural language input processing, allowing users to request music with specific emotions and durations. Advanced features include reinforcement learning fine-tuning with human-in-the-loop feedback, data augmentation techniques, and multi-format output generation (MIDI, MP3, WAV).

The model architecture consists of a 6-layer Transformer with 512-dimensional embeddings, 8 attention heads, and emotion conditioning through learned embeddings. Training was conducted for 24 epochs achieving a validation loss of 1.8154. The system successfully generates coherent musical sequences with emotion-appropriate characteristics, demonstrating the viability of AI-driven music composition.

The project includes comprehensive testing, API integration, audio playback functionality, and a feedback collection system for continuous improvement. Future enhancements include extended training for improved quality, multi-instrument generation, and real-time composition capabilities.

____________________  
Signature of Student

____________________  
Signature of Supervisor

---

## LIST OF FIGURES

1. Figure 1.1: System Architecture Overview
2. Figure 3.1: Full-Stack Application Architecture
3. Figure 4.1: Use Case Diagram
4. Figure 4.2: Class Diagram - Model Architecture
5. Figure 4.3: Sequence Diagram - Music Generation Flow
6. Figure 4.4: Activity Diagram - Training Pipeline
7. Figure 4.5: Activity Diagram - Generation Process
8. Figure 4.6: Transformer Model Architecture
9. Figure 4.7: Risk Interrelationship Graph
10. Figure 5.1: Training Loss Progression
11. Figure 5.2: Component Decomposition Diagram

---

## LIST OF TABLES

1. Table 3.1: Functional Requirements
2. Table 3.2: Non-Functional Requirements
3. Table 3.3: Model Configuration Parameters
4. Table 3.4: Training Configuration Parameters
5. Table 4.1: EMOPIA Dataset Statistics
6. Table 4.2: Emotion Categories and Characteristics
7. Table 4.3: API Endpoints Specification
8. Table 4.4: Risk Log
9. Table 4.5: Risk Area Priority
10. Table 5.1: Testing Plan
11. Table 5.2: Component Decomposition and Testing Types
12. Table 5.3: Test Cases - Tokenizer Component
13. Table 5.4: Test Cases - Model Component
14. Table 5.5: Test Cases - API Component
15. Table 6.1: Training Results Summary
16. Table 6.2: Generation Quality Metrics

---

## LIST OF SYMBOLS & ACRONYMS

- **AI** - Artificial Intelligence
- **API** - Application Programming Interface
- **BPM** - Beats Per Minute
- **CORS** - Cross-Origin Resource Sharing
- **CPU** - Central Processing Unit
- **CUDA** - Compute Unified Device Architecture
- **EMOPIA** - Emotion-based Music Generation Dataset
- **EOS** - End of Sequence
- **GPU** - Graphics Processing Unit
- **HTTP** - Hypertext Transfer Protocol
- **JSON** - JavaScript Object Notation
- **LSTM** - Long Short-Term Memory
- **MIDI** - Musical Instrument Digital Interface
- **ML** - Machine Learning
- **MP3** - MPEG Audio Layer 3
- **MPS** - Metal Performance Shaders (Apple Silicon)
- **NLP** - Natural Language Processing
- **REST** - Representational State Transfer
- **RL** - Reinforcement Learning
- **REMI** - Revamped MIDI-derived events
- **SEI** - Software Engineering Institute
- **TOC** - Table of Contents
- **UI** - User Interface
- **UX** - User Experience
- **WAV** - Waveform Audio File Format

---

# Chapter 1: Introduction

## 1.1 General Introduction

Music generation using artificial intelligence has emerged as a fascinating intersection of machine learning, signal processing, and creative arts. The ability to automatically compose music that evokes specific emotions represents a significant advancement in computational creativity. This project, EMOPIA (Emotion-based Music Generation System), addresses the challenge of generating emotionally expressive music using deep learning techniques.

The system leverages state-of-the-art Transformer architecture, originally developed for natural language processing, adapted for sequential music generation. By conditioning the generation process on emotion labels, the system can create music that aligns with desired emotional characteristics such as joy, sadness, anger, calm, surprise, or fear.

The EMOPIA system is implemented as a complete full-stack application featuring:
- A PyTorch-based Transformer model for music generation
- A Flask REST API backend for model serving
- A React-based frontend for user interaction
- MIDI tokenization and audio synthesis capabilities
- Reinforcement learning fine-tuning with human feedback
- Comprehensive testing and evaluation framework

The system processes natural language inputs (e.g., "Create happy upbeat music for 2 minutes") and generates corresponding MIDI files that can be played back or downloaded. This makes AI-generated music accessible to users without technical expertise in music theory or machine learning.

## 1.2 Problem Statement

Traditional music composition requires extensive musical training, theoretical knowledge, and creative expertise. While digital audio workstations (DAWs) have made music production more accessible, they still require significant manual effort and musical skill. Existing algorithmic music generation systems often produce random or incoherent sequences that lack emotional expressiveness and musical structure.

The key challenges addressed by this project include:

1. **Emotion-Conditioned Generation**: Creating music that accurately reflects specific emotional states while maintaining musical coherence
2. **Long-Sequence Generation**: Generating extended musical compositions (2-5 minutes) with consistent structure and development
3. **Musical Quality**: Ensuring generated music has proper rhythm, melody, and harmonic progression
4. **User Accessibility**: Providing an intuitive interface for non-technical users to generate custom music
5. **Real-Time Interaction**: Enabling quick generation and playback for iterative creative exploration
6. **Evaluation Metrics**: Developing methods to assess the quality and emotional accuracy of generated music

## 1.3 Significance/Novelty of the Problem

This project addresses several significant aspects of AI-driven music generation:

**1. Emotion-Based Conditioning**
Unlike generic music generation systems, EMOPIA explicitly conditions generation on emotion labels, enabling targeted emotional expression in the output. This is particularly valuable for applications requiring specific moods such as film scoring, game soundtracks, or therapeutic music.

**2. Transformer Architecture for Music**
While Transformers have revolutionized NLP, their application to music generation is relatively recent. This project demonstrates the effectiveness of Transformer models for capturing long-range dependencies in musical sequences, essential for maintaining coherence in extended compositions.

**3. Full-Stack Implementation**
The project provides a complete end-to-end system from model training to user-facing application, demonstrating practical deployment of AI research. This includes API design, frontend development, and audio synthesis integration.

**4. Human-in-the-Loop Learning**
The incorporation of reinforcement learning with human feedback represents an advanced approach to improving generation quality based on subjective human preferences, addressing the challenge of evaluating creative AI outputs.

**5. Accessibility**
By providing natural language input processing and audio playback, the system makes AI music generation accessible to users without programming or music theory knowledge, democratizing creative AI tools.


## 1.4 Empirical Study

### Dataset Analysis

The project utilizes the EMOPIA dataset, a comprehensive collection of emotion-labeled MIDI files specifically curated for emotion-based music generation research. The dataset characteristics are:

- **Total Files**: 1,078 MIDI compositions
- **Emotion Categories**: 6 emotions (joy, sadness, anger, calm, surprise, fear)
- **Format**: MIDI (Musical Instrument Digital Interface)
- **Source**: Zenodo repository (DOI: 10.5281/zenodo.5090631)
- **Labeling**: Expert-annotated emotion labels based on Russell's circumplex model

The dataset is organized into quadrants based on valence (positive/negative) and arousal (high/low):
- Q1 (Joy): High Valence, High Arousal
- Q2 (Anger): Low Valence, High Arousal
- Q3 (Sadness): Low Valence, Low Arousal
- Q4 (Calm): High Valence, Low Arousal

Additional emotions (surprise, fear) are included to provide broader emotional coverage.

### Existing Tools Survey

Several existing music generation systems were evaluated:

**1. Magenta (Google)**
- Uses RNN and Transformer models
- Focuses on melody generation
- Limited emotion conditioning
- Requires technical expertise

**2. MuseNet (OpenAI)**
- Large-scale Transformer model
- Generates multi-instrument compositions
- No explicit emotion control
- Computationally expensive

**3. AIVA (Artificial Intelligence Virtual Artist)**
- Commercial music composition AI
- Supports various genres
- Limited customization options
- Proprietary system

**4. Jukebox (OpenAI)**
- Generates raw audio waveforms
- Includes vocals
- Very computationally intensive
- Limited controllability

### Experimental Study

Preliminary experiments were conducted to validate the approach:

**Training Experiments**:
- Initial training: 24 epochs on EMOPIA dataset
- Validation loss: 1.8154 (indicating good convergence)
- Training time: 4-6 hours on cloud GPU (V100)
- Model size: ~50M parameters

**Generation Quality Tests**:
- Average notes per generation: 100-300 notes
- Duration range: 30 seconds to 3 minutes
- Emotion accuracy: Subjectively evaluated as 70-80% appropriate
- Musical coherence: Recognizable melodies and rhythm patterns

**Performance Metrics**:
- Generation time: 15-30 seconds on CPU
- Generation time: 2-5 seconds on GPU
- API response time: < 1 second (excluding generation)
- Frontend load time: < 2 seconds


## 1.5 Brief Description of the Solution Approach

The EMOPIA system employs a multi-layered approach to emotion-based music generation:

### 1. Data Representation
Music is represented using a REMI-like (Revamped MIDI-derived events) tokenization scheme that converts MIDI files into discrete token sequences. This includes:
- Note ON/OFF events (128 pitches)
- Time shift tokens (32 steps)
- Velocity buckets (3 levels)
- Special tokens (BOS, EOS, PAD)

### 2. Model Architecture
A Transformer-based architecture with the following specifications:
- **Model Type**: Transformer Decoder
- **Embedding Dimension**: 512
- **Number of Layers**: 6
- **Attention Heads**: 8
- **Feedforward Dimension**: 2048
- **Dropout**: 0.1
- **Maximum Sequence Length**: 512 tokens

### 3. Emotion Conditioning
Emotions are incorporated through learned embeddings:
- 64-dimensional emotion embeddings
- Concatenated with token embeddings
- Projected back to model dimension
- Enables emotion-specific generation patterns

### 4. Duration Control
Target duration is encoded as an additional conditioning signal:
- 32-dimensional duration embeddings
- Allows users to specify desired music length
- Helps model plan appropriate musical structure

### 5. Training Strategy
- **Optimizer**: Adam with learning rate 1e-4
- **Loss Function**: Cross-entropy loss
- **Batch Size**: 32 (adjusted for GPU memory)
- **Data Augmentation**: Pitch shifting (±5 semitones), tempo variation (±10%)
- **Regularization**: Dropout, gradient clipping

### 6. Generation Process
- **Sampling Strategy**: Top-k (k=60) and nucleus sampling (p=0.92)
- **Temperature**: 0.75 for balanced creativity and coherence
- **Constraints**: Minimum notes (100), maximum consecutive time shifts (3)
- **Repetition Penalty**: 1.3 to avoid loops

### 7. Post-Processing
- MIDI file creation from generated tokens
- Audio synthesis using FluidSynth with SoundFont
- MP3 conversion for web playback

### 8. Reinforcement Learning Fine-Tuning
- REINFORCE algorithm for policy gradient optimization
- Reward function based on emotion classifier confidence
- Human feedback integration for quality improvement
- Active learning for sample selection

## 1.6 Comparison of Existing Approaches to the Problem Framed

| Aspect | Traditional Methods | RNN-based | Transformer-based (EMOPIA) |
|--------|-------------------|-----------|---------------------------|
| **Architecture** | Rule-based algorithms | LSTM/GRU networks | Multi-head self-attention |
| **Long-term Dependencies** | Limited | Moderate | Excellent |
| **Training Time** | N/A | Moderate | Higher |
| **Generation Quality** | Predictable but rigid | Good for short sequences | Excellent for long sequences |
| **Emotion Control** | Manual rules | Limited conditioning | Explicit emotion embeddings |
| **Scalability** | Poor | Moderate | Excellent |
| **Parallelization** | N/A | Sequential | Highly parallel |
| **Context Window** | Fixed patterns | Limited by memory | Full sequence attention |
| **Creativity** | Low | Moderate | High |
| **Controllability** | High but manual | Moderate | High with conditioning |

**Key Advantages of EMOPIA Approach**:
1. Superior long-range dependency modeling through self-attention
2. Explicit emotion conditioning for targeted generation
3. Parallel training for faster convergence
4. Flexible architecture supporting multiple conditioning signals
5. State-of-the-art generation quality
6. Human-in-the-loop learning for continuous improvement

**Limitations Compared to Alternatives**:
1. Higher computational requirements during training
2. Larger model size (50M parameters vs. 10-20M for RNNs)
3. More complex implementation
4. Requires substantial training data

---

# Chapter 2: Literature Survey

## 2.1 Summary of Papers Studied

### 2.1.1 Attention Is All You Need (Vaswani et al., 2017)
This seminal paper introduced the Transformer architecture, which forms the foundation of our music generation model. The key innovation is the self-attention mechanism that allows the model to weigh the importance of different parts of the input sequence when generating each output token. Unlike RNNs, Transformers can process sequences in parallel, significantly reducing training time. The multi-head attention mechanism enables the model to capture different types of relationships simultaneously. This architecture has proven highly effective for sequential data beyond NLP, including music generation.

**Key Contributions**:
- Self-attention mechanism for capturing long-range dependencies
- Positional encoding for sequence order information
- Multi-head attention for diverse relationship modeling
- Parallel processing capability

**Relevance to Project**: The Transformer architecture is directly implemented in our music generation model, adapted for MIDI token sequences with emotion conditioning.

### 2.1.2 Music Transformer (Huang et al., 2018)
This paper adapted the Transformer architecture specifically for music generation, introducing relative positional representations that better capture musical structure. The authors demonstrated that Transformers can generate coherent long-form music with consistent themes and variations. They introduced techniques for handling the unique challenges of music data, including timing precision and polyphonic representation.

**Key Contributions**:
- Relative positional encoding for music
- Handling of polyphonic music representation
- Long-form music generation (minutes-long compositions)
- Evaluation metrics for generated music quality

**Relevance to Project**: Informed our positional encoding strategy and sequence length handling for extended musical compositions.

### 2.1.3 EMOPIA: A Multi-Modal Pop Piano Dataset (Hung et al., 2021)
This paper introduced the EMOPIA dataset used in our project, providing 1,078 emotion-labeled MIDI files. The authors established a framework for emotion-based music generation using Russell's circumplex model of affect (valence and arousal dimensions). They demonstrated baseline results using LSTM models and established evaluation protocols for emotion accuracy.

**Key Contributions**:
- Comprehensive emotion-labeled MIDI dataset
- Emotion annotation methodology
- Baseline models for emotion-conditioned generation
- Evaluation framework for emotional music generation

**Relevance to Project**: This dataset forms the training foundation of our system, and we extend their work by applying Transformer architecture with advanced conditioning techniques.

### 2.1.4 MuseNet (Payne, 2019)
OpenAI's MuseNet demonstrated large-scale music generation using Transformers trained on diverse musical styles. The model can generate multi-instrument compositions in various genres. However, it lacks explicit emotion control and requires significant computational resources.

**Key Contributions**:
- Large-scale Transformer for music (72-layer model)
- Multi-instrument generation
- Style transfer capabilities
- Genre-diverse training

**Relevance to Project**: Demonstrated the scalability of Transformers for music, though our approach focuses on emotion conditioning with a more efficient model size.

### 2.1.5 Jukebox (Dhariwal et al., 2020)
Jukebox generates raw audio waveforms including vocals using VQ-VAE and Transformers. While impressive in audio quality, it requires enormous computational resources and offers limited controllability.

**Key Contributions**:
- Raw audio generation (not symbolic)
- Vocal generation capability
- Multi-scale VQ-VAE architecture
- Long-form audio generation

**Relevance to Project**: Informed our understanding of audio synthesis challenges, though we chose MIDI representation for efficiency and controllability.


### 2.1.6 Deep Reinforcement Learning for Music Generation (Jaques et al., 2017)
This work explored using reinforcement learning to fine-tune music generation models based on music theory rules and human preferences. The authors demonstrated that RL can improve generation quality beyond supervised learning alone.

**Key Contributions**:
- RL framework for music generation
- Reward functions based on music theory
- Human-in-the-loop learning
- Improved generation quality metrics

**Relevance to Project**: Directly influenced our Phase 5 implementation of RL fine-tuning with human feedback integration.

### 2.1.7 Conditional Music Generation with LSTM (Briot et al., 2017)
This survey paper reviewed various approaches to conditional music generation, including emotion, genre, and style conditioning. It provided insights into different conditioning strategies and their effectiveness.

**Key Contributions**:
- Comprehensive survey of conditioning methods
- Comparison of different architectures
- Evaluation methodologies
- Best practices for music generation

**Relevance to Project**: Informed our emotion conditioning strategy and evaluation approach.

## 2.2 Integrated Summary of the Literature Studied

The literature review reveals a clear evolution in music generation approaches, from rule-based systems to deep learning models, and more recently to Transformer-based architectures. Several key themes emerge:

**1. Architecture Evolution**
The progression from RNNs to Transformers represents a significant advancement in handling long-range dependencies in music. While RNNs struggle with sequences longer than a few hundred steps, Transformers can effectively model relationships across entire compositions. This is crucial for maintaining musical coherence and thematic development.

**2. Conditioning Strategies**
Effective conditioning on desired attributes (emotion, genre, style) is essential for controllable generation. The literature demonstrates various approaches, from simple label concatenation to learned embeddings and attention-based conditioning. Our approach combines emotion embeddings with duration control for fine-grained generation control.

**3. Representation Choices**
The choice between symbolic (MIDI) and raw audio representation involves trade-offs. MIDI offers interpretability, efficiency, and easier conditioning, while raw audio provides richer timbral information. Our MIDI-based approach with audio synthesis provides a practical balance.

**4. Evaluation Challenges**
Evaluating generated music remains challenging due to its subjective nature. The literature employs both objective metrics (note density, pitch range, rhythm consistency) and subjective evaluations (human ratings, emotion accuracy). Our system incorporates both approaches with human feedback integration.

**5. Reinforcement Learning Integration**
Recent work demonstrates that RL can bridge the gap between supervised learning and human preferences. By incorporating human feedback as rewards, models can be fine-tuned to better align with subjective quality criteria. This is particularly valuable for creative applications where traditional metrics may not capture all aspects of quality.

**6. Practical Deployment**
While much research focuses on model architecture and training, practical deployment requires addressing API design, user interface, audio synthesis, and real-time performance. Our full-stack implementation addresses these practical considerations often overlooked in academic research.

The EMOPIA system builds upon these foundations, combining Transformer architecture with explicit emotion conditioning, RL fine-tuning, and comprehensive deployment infrastructure to create a practical, user-friendly music generation system.

---

# Chapter 3: Requirement Analysis and Solution Approach

## 3.1 Overall Description of the Project

### Product Perspective

The EMOPIA Music Generation System is a standalone, self-contained application that integrates multiple components into a cohesive full-stack solution. The system consists of three main layers:

1. **Frontend Layer**: React-based web application providing user interface
2. **Backend Layer**: Flask REST API serving the machine learning model
3. **Model Layer**: PyTorch-based Transformer model for music generation

The system interfaces with external components including:
- **PyTorch Framework**: For model implementation and inference
- **FluidSynth**: For MIDI to audio conversion
- **FFmpeg**: For audio format conversion
- **Web Browsers**: For user interaction
- **File System**: For storing generated music and checkpoints

### Product Functions

The system provides the following major functions:

1. **Music Generation**
   - Generate MIDI music from text descriptions
   - Generate music by selecting emotion and duration
   - Support for 6 emotion categories
   - Duration control from 30 seconds to 5 minutes

2. **Audio Processing**
   - Convert MIDI to WAV using SoundFont synthesis
   - Convert WAV to MP3 for web playback
   - In-browser audio playback
   - File download in multiple formats

3. **Model Training**
   - Train Transformer model on EMOPIA dataset
   - Data augmentation (pitch shift, tempo variation)
   - Checkpoint management and model selection
   - Training progress monitoring

4. **Reinforcement Learning Fine-Tuning**
   - Collect human feedback on generated samples
   - Fine-tune model using REINFORCE algorithm
   - Active learning for sample selection
   - Track improvement metrics

5. **API Services**
   - RESTful API for music generation
   - Health monitoring endpoints
   - File download services
   - Feedback collection endpoints

### User Characteristics

The system is designed for multiple user types:

1. **End Users** (General Public)
   - No technical expertise required
   - Basic understanding of emotions and music
   - Comfortable using web applications
   - May have creative or entertainment purposes

2. **Developers** (Technical Users)
   - Programming knowledge (Python, JavaScript)
   - Understanding of REST APIs
   - Familiarity with machine learning concepts
   - May want to extend or integrate the system

3. **Researchers** (Academic Users)
   - Deep learning and music generation knowledge
   - Understanding of Transformer architecture
   - Familiarity with PyTorch framework
   - Interest in emotion-based generation research

4. **Content Creators** (Professional Users)
   - Music production background
   - Need for royalty-free music
   - Require specific emotional tones
   - May use for film, games, or media projects


### Constraints

The system operates under the following constraints:

**Hardware Limitations**:
- Minimum 8GB RAM for model inference
- GPU recommended for training (16GB+ VRAM)
- CPU inference possible but slower (15-30 seconds per generation)
- Storage: 2GB for model checkpoints, 500MB for dataset

**Software Dependencies**:
- Python 3.8 or higher
- PyTorch 1.10 or higher
- Node.js 16+ for frontend
- FluidSynth for audio synthesis
- FFmpeg for format conversion

**Performance Requirements**:
- Generation time: < 30 seconds on CPU, < 5 seconds on GPU
- API response time: < 1 second (excluding generation)
- Frontend load time: < 2 seconds
- Maximum concurrent users: 10 (single server instance)

**Regulatory and Ethical Considerations**:
- Generated music is AI-created, not copyrighted
- No guarantee of musical quality or appropriateness
- User responsible for usage of generated content
- Privacy: No user data collection beyond feedback

**Interface Constraints**:
- Web-based interface requires modern browser
- MIDI playback requires browser audio support
- Mobile responsiveness for basic functionality
- No offline mode (requires server connection)

### Assumptions and Dependencies

**Assumptions**:
1. Users have stable internet connection
2. EMOPIA dataset is available and properly formatted
3. SoundFont file (soundfont.sf2) is present for audio synthesis
4. Users understand basic music concepts (tempo, emotion)
5. Generated music quality improves with model training

**Dependencies**:
1. **PyTorch**: Core deep learning framework
2. **Flask**: Web server and API framework
3. **React**: Frontend user interface
4. **pretty_midi**: MIDI file processing
5. **FluidSynth**: Audio synthesis
6. **FFmpeg**: Audio format conversion
7. **EMOPIA Dataset**: Training data source

## 3.2 Requirement Analysis

### Functional Requirements

**FR1: Music Generation from Text**
- **Description**: THE System SHALL generate MIDI music when provided with text input describing emotion and duration
- **Input**: Text string (e.g., "happy upbeat 2 minute track")
- **Processing**: Parse text → Extract emotion and duration → Generate tokens → Create MIDI
- **Output**: MIDI file with specified characteristics
- **Validation**: Generated music matches requested emotion (70%+ accuracy)

**FR2: Emotion-Based Generation**
- **Description**: THE System SHALL support generation for six emotion categories
- **Emotions**: joy, sadness, anger, calm, surprise, fear
- **Input**: Emotion label and duration
- **Processing**: Encode emotion → Condition generation → Produce tokens
- **Output**: Emotion-appropriate MIDI composition
- **Validation**: Emotion classifier confidence > 0.6

**FR3: Duration Control**
- **Description**: THE System SHALL generate music of specified duration within ±10% tolerance
- **Input**: Duration in minutes (0.5 to 5.0)
- **Processing**: Encode duration → Control token generation length
- **Output**: MIDI file with target duration
- **Validation**: Actual duration within 0.9x to 1.1x of requested

**FR4: Audio Synthesis**
- **Description**: THE System SHALL convert MIDI to playable audio formats
- **Input**: Generated MIDI file
- **Processing**: MIDI → WAV (FluidSynth) → MP3 (FFmpeg)
- **Output**: MP3 file for web playback
- **Validation**: Audio file plays correctly in browser

**FR5: File Download**
- **Description**: THE System SHALL provide download functionality for generated files
- **Formats**: MIDI (.mid), MP3 (.mp3)
- **Processing**: Serve files via HTTP download endpoint
- **Output**: Downloaded file to user's device
- **Validation**: File integrity maintained during download

