function J_h = Jacobian_h(N_ris,ris_vec,k_d,x,theta_g) % size - N_ue×3
J_h1 = 1/sqrt(N_ris)*ris_vec*exp(-1i*k_d*[0:N_ris-1]'*(cos(x(3))-cos(theta_g)));
J_h2 = 1i/sqrt(N_ris)*ris_vec*exp(-1i*k_d*[0:N_ris-1]'*(cos(x(3))-cos(theta_g)));
J_h3 = (x(1)+1i*x(2))/sqrt(N_ris)*(1i*k_d*sin(x(3)))*ris_vec*([0:N_ris-1]'...
    .*exp(-1i*k_d*[0:N_ris-1]'*(cos(x(3))-cos(theta_g))));
J_h = [J_h1,J_h2,J_h3];
end