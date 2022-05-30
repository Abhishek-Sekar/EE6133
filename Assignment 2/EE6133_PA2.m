%Multirate Digital Signal Processing Programming Assignment - 2

%Done By - Abhishek Sekar (EE18B067)

clear; %clearing everything out
clc;

%Getting the audio
path = fullfile('Audio_Files','speech8khz.wav') %path where the audio files are hosted

[x fs]=audioread(path); %speech data or music data depending on file

%Original magnitude spectrum
n_dtft=2^(ceil(log2(length(x)))); %number of point in DTFT
X = fftshift(fft(x,n_dtft));
f_dtft = linspace(-1,1,n_dtft); % freq
figure();
spectrum = plot(f_dtft,abs(X)); %plotting magnitude spectrum
xlabel('<- π w ->');
ylabel('Magnitude');
title('Magnitude spectrum of the Speech signal');
x1 = [0.8 0.6];
y1 = [0.5 0.8];
saveas(spectrum,'speech_spectrum.png','png');

%Designing the prototype filter H_o(z)

rp = 0.01;         % Passband ripple in dB 
rs = 50;          % Stopband ripple in dB
w_p = 0.45*fs/2;     %passband frequency
w_s = 0.55*fs/2;     %stopband frequency
f = [w_p,w_s]; % freq for firpm
a = [1 0];        % Desired amplitudes

dev = [(10^(rp/20)-1)/(10^(rp/20)+1) 10^(-rs/20)]; 
[n,fo,ao,w] = firpmord(f,a,dev,fs); %firpmord produces n = 62. As we want type-2, we go with order 43
h_o = firpm(63,fo,ao,w); %chosen 63 as it gives a very good filter


%plot for the filter magnitude
n_dtft=2^(ceil(log2(length(h_o)))); %number of point in DTFT
H_o = fftshift(fft(h_o,n_dtft));
f_dtft = linspace(-1,1,n_dtft); % freq
figure();
prototype = plot(f_dtft,abs(H_o));  %filter magnitude response
xlabel('<- π w ->');
ylabel('Magnitude');
title('Magnitude spectrum of the equiripple prototype LPF');
dim = [.2 .3 .3 .3];
str = {'wp = 0.45π', 'ws = 0.55π'};
annotation('textbox',dim,'String',str,'FitBoxToText','on');
saveas(prototype,'prototype_Ho.png','png');

%performing 2-polyphase decomposition

h_o_0 = h_o(1:2:end); % h_o_0(n) = h_o(2n)
h_o_1 = h_o(2:2:end); % h_o_1(n) = h_o(2n+1)

%plot for the polyphase components
figure();
h_o_stem = stem(h_o);
xlabel('<- n ->');
ylabel('h(n)');
title('Time Domain plot of the prototype');
saveas(h_o_stem,'prototype_stem.png','png');

indices = linspace(1,length(h_o),length(h_o));

%plotting the 0th 2-polyphase component
figure();
h_o_0_stem = stem(indices(1:2:end),h_o_0);
xlabel('<- n ->');
ylabel('h(n)');
title('Portion of the prototype for 0th 2-polyphase component');
saveas(h_o_0_stem,'polyphase_0_stem.png','png');

%plotting the 1st 2-polyphase component
figure();
h_o_1_stem = stem(indices(2:2:end),h_o_1);
xlabel('<- n ->');
ylabel('h(n)');
title('Portion of the prototype for 1st 2-polyphase component');
saveas(h_o_1_stem,'polyphase_1_stem.png','png');

%Filterbank processing begins !!

%Analysis Filterbank operations

%creating a delay element
z_inv = dfilt.delay;

s = filter(z_inv,x); %x delayed by one unit
s = s(2:end);        %s now starts with index 0
s = [s;x(length(x))]; %as filter cuts off the last element

%downsampling both branches

x_d = downsample(x,2); %downsampling by 2
s_d = downsample(s,2); %downsampling by 2

%output through the analysis filters

t_0 = filter(h_o_0,1,x_d); %first branch
t_1 = filter(h_o_1,1,s_d); %second branch

%output through the inverse DFT matrix

v_d_0 = t_0 + t_1; %first branch
v_d_1 = t_0 - t_1; %second branch


% Synthesis Filterbank operations

%output through the DFT matrix

v_d_0_out = v_d_0 + v_d_1;
v_d_1_out = v_d_0 - v_d_1;

%choosing the synthesis filters

%For the first part
k0 = h_o_1;
k1 = h_o_0;

%For the second part
k_0 = h_o_0;
k_1 = h_o_1;

%output through the synthesis filters

%for the first part
y1 = filter(k0,1,v_d_0_out);
y0 = filter(k1,1,v_d_1_out);

%for the second part
y_1 = filter(k_0,1,v_d_0_out);
y_0 = filter(k_1,1,v_d_1_out);

%implementing the final commutator

y_q1 = zeros(size(x));
y_q2 = zeros(size(x));

y_q1(1:2:end) = y0; %when the commutator is at the bottom {2n}
y_q1(2:2:end) = y1; %when the commutator is at the top    {2n+1}


y_q2(1:2:end) = y_0; %when the commutator is at the bottom {2n}
y_q2(2:2:end) = y_1; %when the commutator is at the top    {2n+1}


%generating the output audio file
audiowrite('speech_q1.wav',y_q1,fs)
audiowrite('speech_q2.wav',y_q2,fs)

%plotting Tzp(w) for the first question

%first computing the DTFTs of the two analysis filters

%filter H_0

n_dtft=2^(ceil(log2(length(h_o)))); %number of point in DTFT
H_0 = fftshift(fft(h_o,n_dtft));


%filter H_1
h_i = h_o; 
h_i(2:2:end) = -h_i(2:2:end); % h_1 = (-1)^n*h_0 pi shifted
n_dtft=2^(ceil(log2(length(h_i)))); %number of point in DTFT
H_1 = fftshift(fft(h_i,n_dtft));
f_dtft = linspace(-1,1,n_dtft); % freq
prototype_1 = plot(f_dtft,abs(H_1));  %filter magnitude response
xlabel('<- π w ->');
ylabel('Magnitude');
title('Magnitude spectrum of the equiripple HPF derived from prototype');
saveas(prototype_1,'prototype_H1.png','png');

%filter G_0
g_o = h_o;
n_dtft=2^(ceil(log2(length(g_o)))); %number of point in DTFT
G_0 = fftshift(fft(g_o,n_dtft));
f_dtft = linspace(-1,1,n_dtft); % freq
prototype_2 = plot(f_dtft,abs(G_0));  %filter magnitude response
xlabel('<- π w ->');
ylabel('Magnitude');
title('Magnitude spectrum of the equiripple LPF at the synthesis side');
saveas(prototype_2,'prototype_G0.png','png');


%filter G_1
g_i = -h_i;
n_dtft=2^(ceil(log2(length(g_i)))); %number of point in DTFT
G_1 = fftshift(fft(g_i,n_dtft));
f_dtft = linspace(-1,1,n_dtft); % freq
prototype_3 = plot(f_dtft,abs(G_1));  %filter magnitude response
xlabel('<- π w ->');
ylabel('Magnitude');
title('Magnitude spectrum of the equiripple HPF at the synthesis side');
saveas(prototype_3,'prototype_G1.png','png');


%resulting Tzp(w)

T_zp = 0.5*(abs(H_0 .^2) + abs(H_1 .^2));
figure();
Tzp = plot(f_dtft,abs(T_zp));  %filter magnitude response
xlabel('<- π w ->');
ylabel('Magnitude');
title('Magnitude spectrum of Tzp(w)');
saveas(Tzp,'Tzp.png','png');

%plots of output audio spectrum
n_dtft=2^(ceil(log2(length(y_q1)))); %number of point in DTFT
Y_q1 = fftshift(fft(y_q1,n_dtft));
f_dtft = linspace(-1,1,n_dtft); % freq
figure();
spectrum_q1 = plot(f_dtft,abs(Y_q1)); %plotting magnitude spectrum
xlabel('<- π w ->');
ylabel('Magnitude');
title('Magnitude spectrum of the output Speech signal for part 1');
x1 = [0.8 0.6];
y1 = [0.5 0.8];
saveas(spectrum_q1,'speech_spectrum_q1.png','png');

n_dtft=2^(ceil(log2(length(y_q2)))); %number of point in DTFT
Y_q2 = fftshift(fft(y_q2,n_dtft));
f_dtft = linspace(-1,1,n_dtft); % freq
figure();
spectrum_q2 = plot(f_dtft,abs(Y_q2)); %plotting magnitude spectrum
xlabel('<- π w ->');
ylabel('Magnitude');
title('Magnitude spectrum of the output Speech signal for part 2');
x1 = [0.8 0.6];
y1 = [0.5 0.8];
saveas(spectrum_q2,'speech_spectrum_q2.png','png');














