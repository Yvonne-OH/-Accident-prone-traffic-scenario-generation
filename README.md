\documentclass[10pt, reqno, letterpaper, twoside]{amsart}
\usepackage[margin=1in]{geometry}

\usepackage{amssymb, bm, mathtools}
\usepackage[usenames,dvipsnames,svgnames,table]{xcolor}
\usepackage[pdftex, xetex]{graphicx}
\usepackage{enumerate, setspace}
\usepackage{float, colortbl, tabularx, longtable, multirow, subcaption, environ, wrapfig, textcomp, booktabs}
\usepackage{pgf, tikz, framed, url, hyperref}
\usepackage[normalem]{ulem}
\usetikzlibrary{arrows,positioning,automata,shadows,fit,shapes}
\usepackage[english]{babel}

\usepackage{microtype}
\microtypecontext{spacing=nonfrench}

\usepackage{algorithm}
\usepackage{algorithmic}


\usepackage{times}
\title{Accident Prone Traffic Trajectory Generation Using SocialVAE}
\author{
Zhanqian Wu [zhanqian@seas],
Bowen Jiang [jbwjoy@seas],
Jie Mei [jiemei@seas],
}

\begin{document}

% \Large %%% Remove this after finished

\begin{abstract}

% We investigated the challenge of generating accident-prone traffic scenarios using SocialVAE to assess and train autonomous vehicles (AVs). SocialVAE incorporates a recurrent neural network architecture with LSTM units to generate plausible future vehicle trajectories. The model employs a backward RNN for navigation pattern inference and attention mechanisms to consider neighboring vehicles' states and interactions. This methodology offers a robust way to simulate challenging traffic situations and assess AV planners under complex conditions, including potential accidents. Our project demonstrates qualitative results in various traffic scenarios like intersections and roundabouts, revealing head-on and rear-end collisions. The framework provides a significant tool for refining AV safety and reliability while also offering a streamlined and accessible approach that can operate on standard PC hardware.

This project focuses on generating accident-prone traffic scenarios using SocialVAE to assess and train autonomous vehicles (AVs). 
SocialVAE uses a timewise variational autoencoder (VAE) to generate future vehicle trajectories, leveraging a recurrent neural network (RNN) architecture with Long Short-Term Memory units. 
By conditioning latent variables at each time step and incorporating a backward RNN for navigation inference, the model captures the uncertainty and multimodality of vehicle decision-making. 
An attention mechanism encodes neighboring vehicles' states and interactions to enhance prediction accuracy. 
Experiments on the INTERACTION dataset validated this framework's ability to refine AV safety and reliability, providing an accessible solution that operates on standard PC hardware.

% \Large %%% Remove this after finished

% Five-six short sentences with very clear sentences that mention what the paper is about and what the results are. Note that an abstract is \emph{not} an introduction, there is no need to talk about the importance of a problem in the abstract; the introduction section is the right place to do this.

% 提一嘴Dataset!

\end{abstract}

\maketitle

% \section{Instructions}

% \textbf{
%     \begin{itemize}
%     \item Your report should be limited to 5 pages, 10pt Times New Roman font, 1 inch margins. You will be penalized for reports longer than 5 pages. References do not count in the page limit.
%     \item Your report is the best way for us to judge your work. Make sure you put in enough effort into the report. You should think of it as writing a short research paper/thesis. If you did not mention something in the report or did not write with enough clarity, it is hard for the instructors (or the reviewers of a publication) to judge the quality of your work.
%     \item You can use \href{https://billf.mit.edu/sites/default/files/documents/cvprPapers.pdf}{https://billf.mit.edu/sites/default/files/documents/cvprPapers.pdf} as guidelines for how to write well.
%     \end{itemize}
% }

% The following is an outline for the report. You can use this as a template but feel free to change things.

\section{Introduction}
% Zhanqian 5/9 revised
In the realm of autonomous vehicle (AV) technology, ensuring the safety and reliability of these systems in diverse and unpredictable traffic conditions remains a paramount challenge. Traditional datasets often fall short in providing the variety of complex, real-world scenarios needed to thoroughly test and train AV systems. This limitation is acute in the case of near-collision events, which are rare yet critical for assessing an AV's ability to handle hazardous situations. 



%
% What is the problem you want to solve. Why is it important.

In this study, we introduce a streamlined approach for generating challenging traffic scenarios to evaluate the robustness of AV systems in complex environments. Central to our methodology is the use of a generative model of traffic movement to assess the plausibility of scenarios during optimization by their likelihood. Our Accident Prone Traffic Trajectory Generation method employs a graph-based SocialVAE, which incorporates a vanilla recurrent neural network (RNN) architecture with Long Short-Term Memory (LSTM) units for sequential predictions. This model is enhanced with latent variables that capture the intricate dynamics of vehicle movements, offering a realistic simulation of demanding traffic situations. Additionally, a backward RNN structure is used to infer navigation patterns from complete trajectories, and an attention mechanism dynamically encodes the states of neighboring vehicles, considering their social interactions. These features allow for a detailed representation of traffic dynamics, facilitating the generation of maps that include trajectories with potential accidents. The generated scenarios are then analyzed for qualitative performance, demonstrating the system's capability to operate efficiently on standard PC hardware.
%便于个人PC训练等等.....
%
\subsection{Contributions}\

\begin{enumerate}
    \item \textbf{Innovative Use of SocialVAE}: Our approach incorporates RNNs equipped with LSTM units~\cite{7780479}. This setup enables the generation of complex traffic scenarios that closely mimic real-world interactions among vehicles.

    \item \textbf{Attention Mechanism for Neighbor Encoding}: An attention mechanism to encode the states of neighboring vehicles considers the social features exhibited by these entities. This development is critical in scenarios with dense traffics, ensuring model accuracy among vehicles in proximity.

    \item \textbf{Practical Implementation on PCs}: The methodology is designed to run on standard PCs, enhancing accessibility and broadening testing capabilities. 
\end{enumerate}
% A list of concrete results in the report that the reader can quickly ascertain/understand.

\section{Background}
%
% Introduce the problem and the notation if any.
Caesar et al. (2020)~\cite{caesar2020nuscenes} and Houston et al. (2020)~\cite{houston2020one} have noted that traditional datasets primarily sourced from real-world driving are significantly limited due to the rarity of near-collision scenarios, which are crucial for testing autonomous vehicle (AV) systems. While simulation platforms like CARLA (Dosovitskiy et al., 2017)~\cite{dosovitskiy2017carla} and NVIDIA’s DRIVE Sim have addressed these issues by providing controlled environments where diverse and uncommon scenarios can be tested, these tools still struggle with replicating the dynamic complexity of real-world conditions.

Innovations by Bergamini et al. (2021)~\cite{bergamini2021simnet} have advanced the field by using deep learning techniques such as variational autoencoders (VAEs) and GANs to generate more plausible and challenging traffic scenarios. Despite these advancements, existing simulations often fail to adjust in response to the evolving behaviors of AV systems during testing, limiting their application in developing robust decision-making frameworks for AVs.

To surmount these challenges, our approach incorporates the SocialVAE~\cite{Xu_2022} to create adaptive traffic scenarios that more effectively test AV systems. This method simulates a broad range of adversarial conditions within a learned traffic model, dynamically generating scenarios that provoke specific undesirable behaviors from the AV. Unlike previous methods, our approach does not rely on a set adversarial strategy; instead, it continuously adapts to the AV’s reactions, ensuring that the scenarios are both realistic and tailored to test the AV’s unique capabilities thoroughly. This technique aims to enhance the safety and reliability of AVs by providing a more comprehensive testing framework that reflects the unpredictable nature of real-world driving.

\section{Related Work}
%
% Note down references. Say how they relate to your approach. Your objective to put your work in context of the broader literature, namely, what other possible approaches exist for this problem, what they are good at or what they lack, what your approach does differently from them.
Xu et al.'s work~\cite{Xu_2022} contributes to understanding pedestrian dynamics within traffic systems using a sophisticated timewise VAE. This focus on pedestrian behavior. However, SocialVAE primarily addresses pedestrian trajectories and does not extend to the intricacies of vehicular dynamics. Our approach enhances the scope of traffic management systems to better predict complex traffic interactions.

On the other hand, Rempe et al.'s research~\cite{strive} emphasizes the generation of challenging vehicular scenarios, employing a graph-based conditional VAE (CVAE) to create challenging traffic conditions. This is pivotal for testing the limits of predictive capabilities under potential collision scenarios. Although it is highly effective , STRIVE's model training and simulation largely depends on hardware requirements. Our approach replaced the CVAE with SocialVAE, which has a simpler configuration. Thus, personal implementation on a PC or laptop becomes available.
\section{Approach}

% Details of your approach.

\subsection{Overall Structure}\

Following previous research~\cite{wang2023advsim}, the scenario generation is approached as an optimization problem that modifies agent trajectories in a baseline scenario derived from real-world data.
The SocialVAE method estimates the distribution of future trajectories for each agent in a scene, using historical observations. It predicts each agent (ego vehicle)'s future independently and can handle scenes with any number of agents. Undesirable outcomes include collisions, uncomfortable driving conditions, and violations of traffic laws. The computation graph shows the state transfer inside the VAE. The overall structure is shown in Fig.~\ref{fig:flowchart}.
\begin{figure}[!htb]
    \centering
    \includegraphics[width=0.65\linewidth]{figs/flowchart.png}
    \caption{An overview of accident prone traffic trajectory generation with SocialVAE, which incorporates a recurrent neural network (RNN)-based VAE operating in a timewise manner with stochastic latent variables generated sequentially for predicting trajectories. The observation encoder's attention mechanism takes into account the state $n_{j|i}$ and social features $k_{j|i}$ of each neighboring entity. The diagram on the right illustrates the flow of states within the timewise VAE.}
    \label{fig:flowchart}
\end{figure}

\subsection{SocialVAE} \

\noindent \textbf{Generative Model:} Using the LSTM Structure,instead of directly predicting the absolute coordinates, we define a displacement sequence \(d_{t+1:t+H}^i\). The generative model is defined as Eq.~\ref{eq:genmodel}, where \(z_t^i\), \(d_t^i\) and \(o_{1:T}^i\) denote the latent variables introduced at time step \(t\), the displacement sequence and the observation sequence, respectively. 
\begin{equation}
p(\mathbf{d}_i^{T+1:T+H}|\mathcal{O}_i^{1:T})=\prod_{t=T+1}^{T+H}\int_{\mathbf{z}_i^t}p(\mathbf{d}_i^t|\mathbf{d}_i^{T:t-1}, \mathcal{O}_i^{1:T},\mathbf{z}_i^t)p(\mathbf{z}_i^t|\mathbf{d}_i^{T:t-1}, \mathcal{O}_i^{1:T})\mathbf{d}\mathbf{z}_i^t.
\label{eq:genmodel}
\end{equation}
%
% \begin{itemize}
%   \item[-] \(z_t^i\) denotes the latent variables introduced at time step \(t\)
%   \item[-] \(d_t^i\) denotes the displacement sequence
%   \item[-] \(o_{1:T}^i\) denotes the observation sequence
% \end{itemize}
To implement the sequential generative model \(p(d_{t+1}^i | o_{1:T}^i, z_t^i)\), we use LSTM where the state variable \(h_t^i\) is updated recurrently by $\mathrm h_i^t=\overrightarrow g(\psi_{\mathbf z\mathrm d}(\mathrm z_i^t,\mathrm d_i^t),\mathrm h_i^{t-1})$, where $t = T+1,..., T+H$. The prior distribution of SocialVAE is conditioned and can be obtained from the LSTM state variable. The second term of Eq.~\ref{eq:genmodel} can be expressed as Eq.~\ref{eq:2ndterm}, where $\theta$ are parameters for a neural network to be optimized.
\begin{equation}
p(\mathbf{z}_i^t|\mathbf{d}_i^{T:t-1}, \mathcal{O}_i^{1:T}):=p_\theta(\mathbf{z}_i^t|\mathbf{h}_i^{t-1})
\label{eq:2ndterm}
\end{equation}

\noindent \textbf{Latent Space Sampling:} The first component of the integral shown in Eq.~\ref{eq:genmodel} suggests that new displacements are sampled from the prior distribution \(p\), which depends on the latent variable \(z_t^i\) and incorporates both observations and earlier displacements as reflected by $h_{t-1}^i$. Thus, $\mathrm d_i^t\sim p_\xi(\cdot|\mathrm z_i^t,\mathrm h_i^{t-1})$ represents the sampled displacement. where $z_{i}^{t}$, $h_{i}^{t-1}$ and $\xi$ denote conditioned latent variables, previous displacements and the observation sequence, respectively. Therefore, we can obtain $\mathbf{x}_i^t=\mathbf{x}_i^T+\sum_{\tau=T+1}^t\mathbf{d}_i^\tau$ as a stochastic estimation for the spatial position at time $t$.
% \begin{equation}
% \mathbf{x}_i^t=\mathbf{x}_i^T+\sum_{\tau=T+1}^t\mathbf{d}_i^\tau
% \label{eq:stodisp}
% \end{equation}

\noindent \textbf{Inference Model:} 
To estimate the posterior distribution \(q\) over the latent variables, the entire GT observation sequence from \(O_{1:T+H}^i\) is utilized. This is denoted by Eq.~\ref{eq:infmodel}, where \(t\) ranges from \(T+1\) to \(T+H\), and the initial state \(b_{T+H+1}^i=0\). The backward state \(b_t^i\) transmits GT trajectory data from \(T+H\) down to \(t\), forming the posterior by combining information from both the backward state \(b_t^i\) and the forward state \(h_t^i\).
\begin{equation}
\mathbf{b}_i^t=\stackrel{\leftarrow}{g}(\mathcal{O}_i^t,\mathbf{b}_i^{t+1})
\label{eq:infmodel}
\end{equation}

\noindent \textbf{Observation Encoding:} If there are multiple neighboring agents in the scene during the prediction process. We need to treat the local observation from agent $i$ to the scene at time $t=2,...,T$ as Eq.~\ref{eq:obsenc}. This includes data from agent and a combined representation of all its neighboring agents.
$s_i^t$ is the self-state of agent $i$, $\mathbf{n}_{j|i}^{t}$ is the local state of neighbor agent $j$, $f_s$, $f_n$ are learnable feature extraction neural networks and $w_j^t|i$ is the attention mechanism weight if $t\leq T$.
\begin{equation}
\mathcal{O}_i^t:=\begin{bmatrix}f_\mathbf{s}(\mathbf{s}_i^t),\sum_jw_{j|i}^tf_\mathbf{n}(\mathbf{n}_{j|i}^t)\end{bmatrix}
\label{eq:obsenc}
\end{equation}

\noindent \textbf{Training Loss:} The VAE calculates the loss for backpropagation and network weight updates. The loss is a combination of several components:$\min_{\theta} \mathcal{L}_\text{kl} + \mathcal{L}_\text{mse} + \mathcal{L}_\text{adv} + \mathcal{L}_\text{kin}$
% \begin{equation}
%     \min_{\theta} \mathcal{L}_\text{kl} + \mathcal{L}_\text{mse} + \mathcal{L}_\text{adv} + \mathcal{L}_\text{kin}
% \end{equation}

\begin{itemize}
    \item[-] \textbf{KL Loss}: Measures the difference between the encoded distribution and a standard normal distribution.
    \begin{equation}
        \mathcal{L}_\text{kl} = w_{KL} D_{KL}\left[q_{\phi}(\mathbf{z}_{i}^{t}|\mathbf{b}_{i}^{t},\mathbf{h}_{i}^{t-1})||p_{\theta}(\mathbf{z}_{i}^{t}|\mathbf{h}_{i}^{t-1})\right]
    \end{equation}
    \item[-] \textbf{Adversarial Loss}: Penalizes predicted trajectories that come too close to neighboring trajectories, using Euclidean distance between the i-th predicted point and the j-th neighbor's position, i.e. $\mathbf{e}_{i,j}=\hat{\mathbf{y}}_i-\mathbf{n}_{i,j}$.
    \begin{equation}
        \mathcal{L}_\text{adv} = \sum_{i=1}^N\sum_{j=1}^M\exp\left(-\sqrt{\|\mathbf{e}_{i,j}\|_2}\right)\cdot\frac1{\sum_{k=1}^M\exp\left(-\sqrt{\|\mathbf{e}_{i,k}\|_2}\right)}
    \end{equation}
    \item[-] \textbf{Average Weighted MSE}: Weighted version of the mean squared error between original and reconstructed data. Let $w_t=\exp(-\alpha t)$ be the weight for time step t, where $\alpha$ is the decay rate. 
    \begin{equation}
        \mathcal{L}_\text{mse} = \frac{\sum_{t=1}^Tw_t\sum_{i=1}^N(\hat{\mathbf{y}}_{t,i}-\mathbf{y}_{t,i})^2}{\sum_{t=1}^Tw_t}
    \end{equation}
    \item[-] \textbf{Kinematic Loss}: Penalizes deviations in velocities and angular velocities of the predicted trajectories, where $\hat{\mathbf{d}}_t$, $\Delta\theta_t$ are the displacement and angular velocity at time $t$.
    \begin{equation}
        \mathcal{L}_\text{kin} = \sum_{t=1}^{T-1}\lVert\hat{\mathbf{d}}_{t+1}-\hat{\mathbf{d}}_t\rVert_2 + \sum_{t=1}^{T-2} \|\Delta\theta_{t+1}-\Delta\theta_t\|_2
        \label{eq:kin_loss}
    \end{equation}
\end{itemize}

\noindent \textbf{Final Position Clustering (FPC):} FPC is implemented to improve the diversity of trajectories. For each cluster, FPC selects the trajectory closest to the center, generating a diverse set of predictions, as shown in Fig.~\ref{fig:fpc}. This approach reduces prediction bias by avoiding the over-representation of trajectories from high-density regions. 

\begin{figure}[!htb]
    \centering
    \includegraphics[width=0.4\linewidth]{figs/fpc.png}
    \caption{An example of FPC to extract 3 predictions from 9 candidates}
    \label{fig:fpc}
\end{figure}

\section{Experiments}

% Make sure you write clearly which datasets/architectures are used, how you pre-process the data, why you are reporting the metrics you are reporting. You should not simply say ``I did this experiment and this is the error'' like you'd do in a problem set. The objective of this section is to interpret the results, explain what they mean, what is good, what is bad etc.

\subsection{Implementation Details} \

\noindent \textbf{Dataset. } 
The INTERACTION Dataset~\cite{interactiondataset} provides a diverse range of global driving scenarios crucial for autonomous vehicle research. It includes detailed annotations of dynamic behaviors and complex interactions, supporting studies in motion prediction and behavior modeling to enhance vehicle safety and efficacy.
% The nuScenes dataset \cite{caesar2020nuscenes} is a large-scale, multimodal collection designed for autonomous driving research, providing rich sensor data and annotations across diverse urban scenarios. It is used both to train our traffic model and to initialize adversarial optimization. We used the same splits and settings as \cite{strive}.

% \noindent \textbf{Planners. } A rule-based planner similar to the lane-graph-based planners from the 2007 DARPA Urban Challenge \cite{planner36, planner51} and adapted by \cite{strive} is used to demonstrate the scenario

% \noindent \textbf{SocialVAE. }  SocialVAE is implemented in replacement of the CVAE in \cite{strive} to predict and generate ego vehicle future trajectories using a tensor of historical trajectories and the observable states of neighboring vehicles. It is initialized with predict horizon, observation radius and hidden layer dimension. The structure is adapted from \cite{Xu_2022}. 
% Alg.~\ref{alg:socialvaeclass} defines the class of the SocialVAE algorithm. 


%%% 这里似乎可以不要
% \begin{algorithm}[!htb]
%     \caption{SocialVAE Structure}
%     \begin{algorithmic}[1]
%     \STATE {\bf class} SocialVAE:
%         \STATE\quad {\bf function} \_\_init\_\_():
%             \STATE\quad\quad Initialize model parameters
%             \STATE\quad\quad Define sub-modules and RNNs
%         \STATE\quad {\bf function} attention(q, k, mask):
%             \STATE\quad\quad Compute \& return attention weights
%         \STATE\quad {\bf function} enc(x, neighbor, y):
%             \STATE\quad\quad Compute social features
%             \STATE\quad\quad Update RNN state
%             \STATE\quad\quad Return final state
%         \STATE\quad {\bf function} forward(x, neighbor, n\_predictions):
%             \STATE\quad\quad\textbf{if} training \textbf{then} Call learn function
%                 \STATE\quad\quad Generate \& return predictions
%         \STATE\quad {\bf function} learn(x, y, neighbor):
%             \STATE\quad\quad Encode inputs
%             \STATE\quad\quad Compute \& return errors and losses
%         \STATE\quad {\bf function} loss(err, kl, L\_adv\_loss, avg\_weighted\_mse\_loss, Knematic\_loss):
%             \STATE\quad\quad Compute \& return total and individual losses
%     \end{algorithmic}
%     \label{alg:socialvaeclass}
% \end{algorithm}

\noindent \textbf{Training. } In \cite{strive}, training was conducted on a computing cluster comprising an NVIDIA Titan RTX GPU and 12 Intel i7-7800X @3.5GHz CPUs, offering significantly greater computational power and memory than a personal computer. We utilized SocialVAE and a smaller dataset, making training feasible on a personal computer, while still achieving favorable results with the generated trajectories within hours. Hardware and parameters we used are listed in Tab.~\ref{tab:socialvae}. Training losses are plotted in Fig.~\ref{fig:losses}. 

\begin{table}[h]
\caption{SocialVAE Training and Hyperparameters}
\label{tab:socialvae}
\centering
\SMALL
\begin{tabularx}{\textwidth}{XXXX} % Four columns
\toprule % Top horizontal line
\multicolumn{4}{c}{\textbf{Hardware}} \\
\midrule
\textbf{Parameter} & \textbf{Value} & \textbf{Parameter} & \textbf{Value} \\
\midrule
Computing Platform & NVIDIA RTX 2060 GPU & CPU & Intel i7-9750H @ 2.6GHz \\
GPU Memory & 6GB & RAM & 16GB \\

\midrule
\multicolumn{4}{c}{\textbf{Hyperparameters}} \\
\midrule
\textbf{Parameter} & \textbf{Value} & \textbf{Parameter} & \textbf{Value} \\
\midrule
Utilized Model & SocialVAE & Observation Radius & 10000 \\
Prediction Time Steps & 25 & Observation Time Steps & 10 \\
RNN Hidden Layer Dim & 512 & Latent Variable Dim & 32 \\
Embedding Layer Dim & 128 & Input Dim & 2 \\
Feature Dim & 256 & Batch Size & 128 \\
Learning Rate & $1 \times 10^{-4}$ & Weight Scaling Factor & 0.1 \\
\bottomrule % Bottom horizontal line
\end{tabularx}
\end{table}


% \begin{figure}
%     \centering
%     \includegraphics[width=0.5\linewidth]{figs/traintrain_rec.png}
%     \caption{Enter Caption}
%     \label{fig:enter-label}
% \end{figure}

\begin{figure}[!htb]
    \centering
    \includegraphics[width=1.0\linewidth]{figs/losses.png}
    \caption{Losses}
    \label{fig:losses}
\end{figure}

% \begin{figure}[!htb]
%     \centering
%     \begin{subfigure}[t]{0.24\textwidth}
%     \centering
%     \includegraphics[width=\linewidth]{figs/avr_mse_loss.png}
%     \label{fig:ppt_p5_1}
%     \caption{Generated}
%     \end{subfigure}
%     \begin{subfigure}[t]{0.24\textwidth}
%     \centering
%     \includegraphics[width=\linewidth]{figs/kl_loss.png}
%     \label{fig:ppt_p5_2}
%     \end{subfigure}
%     \begin{subfigure}[t]{0.24\textwidth}
%     \centering
%     \includegraphics[width=\linewidth]{figs/adv_loss.png}
%     \label{fig:ppt_p6_1}
%     \end{subfigure}
%     \begin{subfigure}[t]{0.24\textwidth}
%     \centering
%     \includegraphics[width=\linewidth]{figs/train_loss.png}
%     \label{fig:ppt_p6_2}
%     \end{subfigure}
%     \caption{Generated (\#\#\# Placeholder)}
%     \label{fig:ppt_figures}
% \end{figure}



%% Data are from the config files (interaction)
% \begin{table}[h]
% \centering
% \begin{tabular}{cc}
% \hline
% \textbf{Hyperparameters} & \textbf{Value} \\
% \hline
% Neighbor observation radius & 10000 \\ % ob\_radius
% Time steps for prediction & 25 \\ % pred\_horizon
% Time steps for observation & 10 \\ % ob\_horizon
% Dimension of RNN hidden layers & 512 \\ % hidden\_dim default 256
% Dimension of latent variables & 32 \\ % z\_dim
% Dimension of embedding layers & 128 \\ % embed\_dim
% % Dimension of output layers & \\ % output\_dim
% Input dimension & 2 \\ % d\_dim
% Dimensions of features & 256 \\ % feature\_dim &
% Batch size & 128 \\
% Learning rate & $1\times10^{-4}$ \\
% Weight scaling factor & 0.1 \\ % alpha
% \hline
% \end{tabular}
% \caption{Hyperparameters of SocialVAE}
% \label{tab:hyperparams}
% \end{table}



% \noindent \textbf{Metrics. } Similar to \cite{strive}, we defined \textit{collision rate} as the fraction of optimized initial scenarios from nuScenes that succeed in causing a planner collision, which indicates the sample efficiency of scenario generation. \textit{Losses} including training adversarial loss ($\mathcal{L}_{adv}$), weighted average MSE loss ($\mathcal{L}_{mse}$) and total training loss are also used to indicate training outline.

\subsection{Qualitative Results of Scenario Generation} \

Compared to rolling out a given planner on unmodified scenarios, challenging scenarios from our approach produce collisions and less comfortable driving. Fig.~\ref{fig:usa_intersection_ep1} presents the generated trajectories of the ego vehicle in the Intersection EP1 scenario, depicting head-on collisions and rear-end collisions with surrounding vehicles in the top and bottom rows, respectively. Fig.~\ref{fig:usa_roundabout_ft} illustrates the generated trajectory of the ego vehicle in a head-on collision with a pedestrian and the trajectory of the ego vehicle side-impacting surrounding vehicles in the USA Roundabout FT scenario.

% \begin{figure}[!htb]
%     \centering
%     \begin{subfigure}[t]{0.45\textwidth}
%     \centering
%     \includegraphics[width=\linewidth]{figs/ppt_p5_1.png}
%     \label{fig:ppt_p5_1}
%     \caption{Generated}
%     \end{subfigure}
%     \begin{subfigure}[t]{0.45\textwidth}
%     \centering
%     \includegraphics[width=\linewidth]{figs/ppt_p5_2.png}
%     \label{fig:ppt_p5_2}
%     \end{subfigure}
%     \begin{subfigure}[t]{0.45\textwidth}
%     \centering
%     \includegraphics[width=\linewidth]{figs/ppt_p6_1.png}
%     \label{fig:ppt_p6_1}
%     \end{subfigure}
%     \begin{subfigure}[t]{0.45\textwidth}
%     \centering
%     \includegraphics[width=\linewidth]{figs/ppt_p6_2.png}
%     \label{fig:ppt_p6_2}
%     \end{subfigure}
%     \caption{Generated (\#\#\# Placeholder)}
%     \label{fig:ppt_figures}
% \end{figure}

\begin{figure}[!htb]
    \centering
    \includegraphics[width=0.75\linewidth]{figs/Generated EP1.png}
    \caption{Generated Trajectory in Intersection Scenario}
    \label{fig:usa_intersection_ep1}
\end{figure}

\begin{figure}[!htb]
    \centering
    \includegraphics[width=0.75\linewidth]{figs/Generated Roundabout FT.png}
    \caption{Generated Trajectory in Roundabout Scenario}
    \label{fig:usa_roundabout_ft}
\end{figure}

\section{Discussion}

% This section explains how to interpret the results in the context of the broader literature. You can talk about what did not work, what you'd like to do if you have more time/data/resources or more long-term investment that one would need to do to solve this specific problem.


By integrating the SocialVAE architecture into STRIVE's framework, our project successfully generated complex traffic accident scenarios for testing AV planners. 
SocialVAE's timewise latent variable generation and attention mechanism achieved good fidelity of accident-prone scenarios.
This integration brings forth more challenging scenarios that better reflect real-world traffic interactions, allowing rigorous evaluation and refinement. 
Good generation outcome using simplified structure and dataset also provide promising potential for implementing the SocialVAE approach with complete settings as STRIVE.
This project enables autonomous vehicle systems to address a broader spectrum of potential risks, improving their robustness and safety. 

Further explorations for this project includes: 
(1) \textbf{Quantitative analysis and comparison.} Although our experiments produced promising collision trajectories, further quantitative evaluation of the generation process and trajectory quality is still needed, especially compared to the results in \cite{strive, strivecite1, strivecite12}. Future research could employ the same datasets as these papers, 
such as using rule-based planners and larger network.
% utilize rule-based planners in a full framework, use larger network structures, and conduct experiments on more advanced computing platforms.
(2) \textbf{Constraint on trajectories to improve collision rate.} During the generation process, some trajectories didn't result in effective collisions due to the low collision likelihood in the original scenarios and partly because some trajectories prematurely terminate due to distortion before a collision occurs. By designing constraints, we can reduce the manual screening required after generation.
(3) \textbf{More diverse and complex scenarios.} In real-world traffic scenarios, significant numbers of non-motorized participants (e.g., cyclists, pedestrians) also demand careful attention from autonomous driving planners. Future research can incorporate these elements, like temporary changes in road topology, to better refine planners' performance under extreme conditions.

% \section*{References}
\clearpage
\bibliographystyle{plain}
\bibliography{references}

\end{document}
