clear, clc
%% 1 GENERATE CHANNEL PARAMETERS
% 1.1 Basic parameters
SIMULATION_ITER = 1000; % number of iterations of simulation
N_time_slot = 100; % number of time slot
N_bs = 16; % number of BS's antenna
N_ris = 16; % number of RIS's element
N_ue = 1; % number of UE's antenna
d = 0.5; % $d normalized antenna/RIS element spacing \triangleq d_i/lambda \forall i \in {bs, ris, ue}$
k_d = 2*pi*d; % $k_d = \frac{2*\pi}{\lambda}d_i$
SNR = 10^(0/10); % linear sacle - 10dB
sigma_v = 1/sqrt(SNR); % Noise covariance

% 1.2 BS-RIS Channel
theta_g = pi/6;

% 1.3 RIS Reflect Vector
ris_vec = @(x) exp(1i*k_d*x.*[0:N_ris-1]);
x_opt = @(x) (cos(x)-cos(theta_g));

% 1.4 RIS-UE Channel
rho = 0.995; % correlation coefficient of the first-order Gauss-Markov model
% Evolution equation of the first-order Gauss-Markov model
% AoA of UE and AoD of RIS vector via time slot
phi_h = zeros(1,N_time_slot);
phi_h(1) = pi/3; % initial AoD of RIS = 60°

% 1.5 BS-RIS-UE Channel, s = 1, size - 1×1
h = @(x) (x(1)+1i*x(2))/sqrt(N_ris)*ris_vec(0.3)*exp(-1i*k_d*[0:N_ris-1]'*(cos(x(3))-cos(theta_g)));

% 1.6 EKF Parameters
sigma_phi_h = 0.5*pi/180; % sigma_{phi_h}^2 = (0.5°)^2
Q = diag([(1-rho^2)/2,...
    (1-rho^2)/2,...
    sigma_phi_h^2]); % Process noise covariance matrix, size - 3×3
sqrt_Q = sqrt(Q); % size - 3×3
F = diag([rho, rho, 1]); % State transition matrix, size - 3×3
x = zeros(3, N_time_slot); % State vector, size - 3×N_time_slot
y = zeros(N_ue, N_time_slot); % Measurement, size - 1×N_time_slot
err_phi_h = zeros(SIMULATION_ITER,N_time_slot); % estimation error of phi_h


%% 2 EKF TRACKING
for iter = 1:SIMULATION_ITER
    alpha = sqrt(1/2)*(randn + 1i*randn);
    x(:,1) = [real(alpha); imag(alpha); phi_h(1)]; % size - 3×1
    % Generate state transition
    for i = 2:N_time_slot
        x(:,i) = F*x(:,i-1) + sqrt_Q*randn(3,1); % size - 3×1
    end
    % Generate measurement noise
    noise = sqrt(1/2)*(randn(N_ue,N_time_slot) + 1i*randn(N_ue,N_time_slot))/sqrt(SNR);
    x_hat_EKF = zeros(size(x)); % EKF estimation
    x_hat_EKF(:,1) = x(:,1) + sqrt_Q*randn(3,1);
    P_k = Q; % size - 3×3
    for i = 2:N_time_slot
        
        x_hat_k = F*x_hat_EKF(:,i-1); % Predict, size - 3×1
        y(:,i) = h(x(:,i)) + noise(:,i); % Measurement, size - N_ue×1
        J_h = Jacobian_h(N_ris, ris_vec(0.3), k_d, x_hat_k, theta_g); % size - N_ue×3
        y_tilde = [real(y(:,i)); imag(y(:,i))]; % size - 2×N_ue×1
        J_h_tilde = [real(J_h); imag(J_h)]; % size - 2×N_ue×3
        K = P_k*J_h_tilde'/(J_h_tilde*P_k*J_h_tilde'+sigma_v^2/2*eye(2*N_ue)); % size - 3×(2×N_ue)
        h_est = h(x_hat_k); % Estimation
        x_hat_EKF(:,i) = x_hat_k + K*(y_tilde - [real(h_est);imag(h_est)]);
        P_k_k = (eye(3) - K*J_h_tilde)*P_k;
        P_k = F*P_k_k*F' + Q;
    end
    err_phi_h(iter,:) = abs(x(3,:) - x_hat_EKF(3,:)); % abs error of phi_h
end


%% PLOT
MSE_phi_h = mean(err_phi_h.^2); % MSE of phi_h
figure(1)
plot(1:N_time_slot,MSE_phi_h,'r','LineWidth',1.5)
lgd_str = {'AoD MSE'};
xlabel('Time index', 'FontName', 'Times')
ylabel('$\bf{E}\left[ \left| \hat{\phi}-\phi \right|^2 \right] $', 'Interpreter', 'latex')
set(gca,'FontName','Times','FontSize',14,'LineWidth',1.5);
legend(lgd_str, 'FontSize', 14);