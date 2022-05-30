% Multirate Signal Processing Assignment-1 

clear; %clearing everything out
clc;

%To view each plot, comment all previous plots out

%question 1
% Downsampling without AA filtering

path = fullfile('Audio_Files','music16khz.wav') %path where the audio files are hosted

[x fs]=audioread(path); %speech data or music data depending on file

%Original magnitude spectrum
n_dtft=2^(ceil(log2(length(x)))); %number of point in DTFT
X = fftshift(fft(x,n_dtft));
f_dtft = linspace(-1,1,n_dtft); % freq
figure();
q1_plt1 = plot(f_dtft,abs(X)); %plotting magnitude spectrum
xlabel('<- π w ->');
ylabel('Magnitude');
title('Magnitude spectrum of the speech signal');
x1 = [0.8 0.6];
y1 = [0.5 0.8];
annotation('textarrow',x1,y1,'String','Peak')
saveas(q1_plt1,'q1_plt1.png','png');


%Downsampling signal by 2
x_d = downsample(x,2);
n_dtft = 2^(ceil(log2(length(x_d))));
f_dtft = linspace(-1,1,n_dtft);
X_d = fftshift(fft(x_d,n_dtft));

figure();
q1_plt2 = plot(f_dtft,abs(X_d));  %downsampled plot
xlabel('<- π w ->');
ylabel('Magnitude');
title('Magnitude spectrum of the DS-by-2 speech signal');
x1 = [0.8 0.65];
y1 = [0.5 0.7];
annotation('textarrow',x1,y1,'String','Peak')
saveas(q1_plt2,'q1_plt2.png','png');


%generating audio file
audiowrite('music_ds_by2.wav',x_d,fs/2) 


%question 2
% Downsampling by 2 using an AA filter

%creating the required LPF/AA filter

rp = 0.01;         % Passband ripple in dB 
rs = 20;          % Stopband ripple in dB
fs = 16000;        % Sampling frequency, 16kHz for music
%f = [3600 4400];    % LPF cutoff for question 2
f = [1760,2240]; % use this and comment the above for question 3
a = [1 0];        % Desired amplitudes

dev = [(10^(rp/20)-1)/(10^(rp/20)+1) 10^(-rs/20)]; 
[n,fo,ao,w] = firpmord(f,a,dev,fs);
b = firpm(n,fo,ao,w);
n_dtft = 1024
f_dtft = linspace(-1,1,n_dtft);
B = fftshift(fft(b,n_dtft));


%plot for the filter magnitude

figure();
q1_plt_3 = plot(f_dtft,abs(B));  %filter magnitude response
xlabel('<- π w ->');
ylabel('Magnitude');
title('Magnitude spectrum of the equiripple LPF');
dim = [.2 .3 .3 .3];
str = {'wp = 0.45π', 'ws = 0.55π'};
annotation('textbox',dim,'String',str,'FitBoxToText','on');
saveas(q1_plt_3,'q1_plt_3.png','png');


%Passing the input signal through the AA Filter above,

x_aa  = filter(b,1,x) 
x_aad = downsample(x_aa,4) %downsample by 2 for question 2, downsample by 4 for question 5
n_dtft = 2^(ceil(log2(length(x_aad))));
f_dtft = linspace(-1,1,n_dtft);
X_aad = fftshift(fft(x_aad,n_dtft));

%magnitude spectrum of signal after AA and DS by 2
figure();
q1_plt3 = plot(f_dtft,abs(X_aad));  %downsampled plot
xlabel('<- π w ->');
ylabel('Magnitude');
title('Magnitude spectrum of the DS-by-2 speech signal with AA filter');
x1 = [0.8 0.65];
y1 = [0.5 0.7];
annotation('textarrow',x1,y1,'String','Peak')
saveas(q1_plt3,'q1_plt3.png','png');

%generating audio file
audiowrite('music_ds_by2_aa.wav',x_aad,fs/2) %name the audio file appropriately

%question 4
x_u = upsample(x,3)
n_dtft = 2^(ceil(log2(length(x_u))));
f_dtft = linspace(-1,1,n_dtft);
X_u = fftshift(fft(x_u,n_dtft));

%plot of upsampled input
figure();
q1_plt4 = plot(f_dtft,abs(X_u));  %upsampled plot
xlabel('<- π w ->');
ylabel('Magnitude');
title('Magnitude spectrum of the US-by-3 speech signal');
x1 = [0.85 0.8];
y1 = [0.4 0.6];
annotation('textarrow',x1,y1,'String','Peak-3')
x1 = [0.2 0.25];
y1 = [0.4 0.6];
annotation('textarrow',x1,y1,'String','Peak-1')
x1 = [0.6 0.55];
y1 = [0.4 0.6];
annotation('textarrow',x1,y1,'String','Peak-2')
saveas(q1_plt4,'q1_plt4.png','png');

x_u_aa   = filter(b,1,x_u) %passing the upsampled output into AA filter
x_u_aa_d = downsample(x_u_aa,4) %DS by 4
n_dtft = 2^(ceil(log2(length(x_u_aa_d))));
f_dtft = linspace(-1,1,n_dtft);
X_u_aa_d = fftshift(fft(x_u_aa_d,n_dtft));
 %plot of final output
figure();
q1_plt5 = plot(f_dtft,abs(X_u_aa_d));  % plot
xlabel('<- π w ->');
ylabel('Magnitude');
title('Magnitude spectrum of the output speech signal');
x1 = [0.8 0.65];
y1 = [0.5 0.7];
annotation('textarrow',x1,y1,'String','Peak')
saveas(q1_plt5,'q1_plt5.png','png');

%generating audio file
audiowrite('music_u_d.wav',x_u_aa_d,3*fs/4) 

%Question 5
% Go to question 2 make the appropriate changes

x_aad_u = upsample(x_aad,3)
n_dtft = 2^(ceil(log2(length(x_aad_u))));
f_dtft = linspace(-1,1,n_dtft);
X_aad_u = fftshift(fft(x_aad_u,n_dtft));
%plot of the upsampled output after DS by 4

figure();
q1_plt6 = plot(f_dtft,abs(X_aad_u));  %upsampled plot
xlabel('<- π w ->');
ylabel('Magnitude');
title('Magnitude spectrum of the US after DS speech signal');
saveas(q1_plt6,'q1_plt6.png','png');


h = intfilt(3,3,1); %interpolation filter after upsampling
n = linspace(-8,8,length(h));
figure();
q1_plt9 = plot(n,h);  %impulse response plot
xlabel('<- n ->');
ylabel('h[n]');
title('Impulse response of the interpolation filter');
saveas(q1_plt9,'q1_plt9.png','png');

H = fftshift(fft(h, 17));
f_dtft = linspace(-8,8,length(H));
figure();
q1_plt_9 = plot(f_dtft, abs(H));
xlabel("pi w");
ylabel("Magnitude");
title("Magnitude response of interpolation filter");
saveas(q1_plt_9,'q1_plt_9.png','png');

x_out = filter(h,1,x_aad_u) %interpolation
n_dtft = 2^(ceil(log2(length(x_out))));
f_dtft = linspace(-1,1,n_dtft);
X_out = fftshift(fft(x_out,n_dtft));
 %plot of the final output after interpolation
 
%q1_plt7 = plot(f_dtft,abs(X_out));  %magnitude response plot
xlabel('<- π w ->');
ylabel('Magnitude');
title('Magnitude spectrum of the output speech signal after interpolation');
dim = [.2 .3 .3 .3];
str = {'Image',' Rejection'};
annotation('textbox',dim,'String',str,'FitBoxToText','on');
%saveas(q1_plt7,'q1_plt7.png','png');

%generating audio file
audiowrite('music_d_u.wav',x_out,3*fs/4) %name the audio file appropriately











