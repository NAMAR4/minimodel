function new_stim(fname, imaging, fname_protocol, drive_tiff)

filtermode = 0; % normally 1

if isempty(drive_tiff)
    drive_tiff = 'D';
end
% imaging = 1 to trigger the microscope 
% save_flag = 1 to save the timeline variable (takes a bit of time)

global TL

if isempty(imaging)
    imaging = 1;
end

save_flag = 1;
if ~imaging
    disp('THIS IS A TEST:  NOT TRIGGERING THE MICROSCOPE AND NOT SAVING THE TIMELINE')    
    save_flag = 0;
end

froot = 'D:\STIMULUS';
dat = load(fullfile(froot, 'img', fname));

TL = [];
TL.stim.cond_run = 0;

TL.imaging = imaging;
TL.save_flag = save_flag;


x = instrfindall;
for j =1:length(x)
   fclose(x(j));
end
daqreset;

TL.stim.fname = fname;
TL.stim.protocol = fname_protocol;
TL.stim.img   = double(dat.img)/255;

rng(1); % this should really have no effect, but who knows
TL.stim.istim = dat.istim;
TL.stim.nstim = numel(TL.stim.istim);

% default is 7.5 Hz presentation
TL.stim.stim_flips = 4 * ones(TL.stim.nstim,1, 'uint32');
TL.stim.grey_flips = 4 * ones(TL.stim.nstim,1, 'uint32');

TL.stim.cond_microscope = zeros(TL.stim.nstim,1);

if isempty(fname_protocol)    
    % this is 3 Hz presentation, unlinked to microscope frames
%     TL.stim.stim_flips = 19 * ones(TL.stim.nstim,1, 'uint32');
%     TL.stim.grey_flips = 18 * ones(TL.stim.nstim,1, 'uint32');

    % every frame is shown once
    % TL.stim.stim_flips = 1 * ones(TL.stim.nstim,1, 'uint32');
    % TL.stim.grey_flips = 0 * ones(TL.stim.nstim,1, 'uint32');

    % TL.stim.istim = linspace(1, 20, 20*60+1);
    % TL.stim.istim = linspace(1, 15, 20*60+1);
    % TL.stim.istim = TL.stim.istim(1:end-1);    
    % TL.stim.istim = [TL.stim.istim fliplr(TL.stim.istim)];
    % TL.stim.istim = [TL.stim.istim TL.stim.istim(1:2:end)];    
    %TL.stim.istim = repmat(TL.stim.istim, [1, 1000]);
    
    %TL.stim.stim_flips = 1 * ones(TL.stim.nstim,1, 'uint32');
    %TL.stim.grey_flips = 1 * ones(TL.stim.nstim,1, 'uint32');
    
    
   
%     TL.stim.stim_flips = 1 * ones(TL.stim.nstim,1, 'uint32');
%     TL.stim.grey_flips = uint32(poissrnd(2, TL.stim.nstim,1));

    TL.stim.nstim = numel(TL.stim.istim);
else
    dpro = load(fullfile(froot, 'protocol', fname_protocol));
        
    if isfield(dpro, 'istim')
        TL.stim.istim = dpro.istim;
    end
    TL.stim.nstim = numel(TL.stim.istim);
    TL.stim.cond_microscope = zeros(TL.stim.nstim,1);
    
    if isfield(dpro, 'stim_flips')
        TL.stim.stim_flips = ones(TL.stim.nstim,1, 'uint32');
        TL.stim.grey_flips = ones(TL.stim.nstim,1, 'uint32');

        TL.stim.stim_flips(:) = dpro.stim_flips;
        TL.stim.grey_flips(:) = dpro.grey_flips;
    end

    if isfield(dpro, 'dmove')
        TL.stim.tmove = dpro.tmove;
        TL.stim.dmove = dpro.dmove;
    end

    if isfield(dpro, 'cond_microscope')
        if numel(dpro.cond_microscope)==1
            TL.stim.cond_microscope(:) = dpro.cond_microscope;
        else
            TL.stim.cond_microscope = dpro.cond_microscope;            
        end
        disp(size(TL.stim.cond_microscope))
    end

    if isfield(dpro, 'cond_run')
        TL.stim.cond_run = dpro.cond_run;
    end
end

% TL.cond_microscope = 0;

if TL.imaging
    msg = 'instruction hello ';
    u = udp('10.123.1.155', 1002, 'LocalPort', 1001);
    fopen(u);

    fwrite(u, msg);

    TL.sess.fs      = str2double(char(fread(u)));
    TL.sess.nplanes = str2double(char(fread(u)));

    if isnan(TL.sess.fs)
        error('Could not communicate with scanimage computer');
    end
end

TL.stop_flag = 0;

TL.stim.iscreen = 0;
TL.stim.frame = [];
TL.stim.ind_stim = 0;
TL.stim.last_stimulus = now;
TL.stim.data = [];
TL.stim.colormask = [1 0 .5]; % maximum is 1 1 1

% TL.beh.run_now = 0;
% TL.beh.run_speed = 0;

%%%%%%%% add by Lin
TL.beh.run_speed = zeros(3600*4*60,1);
TL.beh.run_now = zeros(3600*4*60,1);

TL.beh.irun = 0;
TL.beh.eye_position = 0;
TL.beh.run_th = 0.01;
TL.beh.iface = 0;

TL.daq.nframes = 0;
TL.daq.last_mic = now;
TL.daq.ipack = 0;
TL.daq.pack = [];

TL.sess.srate = 5000;
root    = 'D:\EXP';
datexp = datestr(now, 'yyyy_mm_dd');
% input animal name

if save_flag
    while 1
        A = input('what is the animal name?', 's');
        if ~isempty(A)
            break;
        end
    end
else
    A = 'test';
end

mname = A;
fpath = fullfile(root, mname, datexp);
fld = dir(fpath);
nmax = 0;
for i = 3:numel(fld)
   if fld(i).isdir
       nmax = max(nmax, str2num(fld(i).name));
   end
end
blockexp = sprintf('%d', nmax+1);
fdata= fullfile(fpath, blockexp);
mkdir(fdata);
fprintf('Block %s, of mouse %s, date %s \n', blockexp, mname, datexp);

TL.myMQTT = [];

TL.myMQTT = mqttclient('tcp://127.0.0.1');
fpath = fullfile('D:\EXP', mname, datexp, sprintf('%s', blockexp));
write(TL.myMQTT, 'trigger', fullfile(fpath, fname));
disp('sent trigger to facemap')
subscribe(TL.myMQTT, 'pupil/online');
%subscribe(TL.myMQTT,'sphericalTreadmill/Data','QualityOfService',0);

TL.beh.s_port = connect_to_treadmil();
data = read_treadmill(TL.beh.s_port);
TL.beh.last_dist = data(2);

pause(.1);

%mqttMsg = read(TL.myMQTT, Topic = 'sphericalTreadmill/Data');
%if isempty(mqttMsg)
%    error('cannot communicate with ball')
%end
%[rpy, rpy_all] = parse_ball(mqttMsg);

%disp('connection with ball established')


% Psych toolbox setup
sca;
close all;
PsychDefaultSetup(2);
Screen('Preference', 'SkipSyncTests', 1);
Screen('Preference','VisualDebugLevel', 0);
% screens = Screen('Screens');

screenNumber = get_ipad_number();
disp(screenNumber)
nscreens = 3;
calib = get_calib(1536, 2048, nscreens, screenNumber);
PsychImaging('PrepareConfiguration');
PsychImaging('AddTask', 'AllViews', 'GeometryCorrection', calib);

[TL.stim.window, TL.stim.destRect] = PsychImaging('OpenWindow', screenNumber, ...
    .5 * TL.stim.colormask);
Screen('BlendFunction', TL.stim.window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

TL.stim.dshift = TL.stim.destRect;
Screen('FillRect', TL.stim.window, .5 * TL.stim.colormask);
Screen('AsyncFlipBegin', TL.stim.window, 0, 0);


root_data = ['' ...
    '' ...
    '' ...
    '' ...
    sprintf('%s:\\DATA\\', drive_tiff)];
fname = sprintf('%s_%s_%s', mname, datexp, blockexp);
fpath = fullfile(root_data, mname, datexp, sprintf('%s', blockexp), fname);
msg = sprintf('instruction ExpStart filepath %s', fpath);

TL.sess.sread = daq.createSession('ni');
addAnalogInputChannel(TL.sess.sread,'Dev4',1,'Voltage'); % read in the twop
TL.sess.sread.Rate = TL.sess.srate;
TL.sess.sread.IsContinuous = true;

TL.sess.readL = addlistener(TL.sess.sread,'DataAvailable',@acquire_TL);
TL.sess.sread.NotifyWhenDataAvailableExceeds = 80;

TL.sess.mname = mname;
TL.sess.datexp = datexp;
TL.sess.fdata = fdata;
TL.sess.blockexp = blockexp;

startBackground(TL.sess.sread); % start reading frames

TL.stim.nflip = 1;

pause(0.25);

if TL.imaging
    fwrite(u, msg);
    pause(6);
end

tic;
while TL.stim.ind_stim < TL.stim.nstim % toc < 30
    % check for key press every time
    [a,b,keyCode] = KbCheck;
	if keyCode(120)>0 || (TL.stop_flag ==1)
		break;
	end

    % check if the screen has flipped
    [VBLTimestamp, StimulusOnsetTime] = Screen('AsyncFlipCheckEnd', TL.stim.window);
    if VBLTimestamp <1e-5        
        pause(0.1/60);
        continue;
    end

    TL.stim.iscreen = TL.stim.iscreen + 1; 
    if TL.stim.nflip==0
        TL.stim.last_stimulus = now;
        TL.stim.data(TL.stim.ind_stim).time    = TL.stim.last_stimulus;
        TL.stim.data(TL.stim.ind_stim).iscreen = TL.stim.iscreen;
    end
    TL.stim.nflip = TL.stim.nflip + 1;    
    
    % get the running speed
%     mqttMsg = read(TL.myMQTT, Topic = 'sphericalTreadmill/Data');
%     [rpy, rpy_all] = parse_ball(mqttMsg);
%     TL.beh.irun = TL.beh.irun + 1;
%     TL.beh.running(TL.beh.irun).rpy(:,1)  =  rpy;
%     TL.beh.running(TL.beh.irun).time =  now;
%     TL.beh.run_now = -rpy(2);    
%     TL.beh.run_speed = exp(-.2/60) * TL.beh.run_speed + ...
%         (1 - exp(-.2/60)) * TL.beh.run_now;

    % get the pupil position and area
    mqttMsg = read(TL.myMQTT, Topic = 'pupil/online');
    for j = 1:size(mqttMsg,1)        
        pup = sscanf(mqttMsg(j,2).Data{1}, '%f');
        TL.beh.iface = TL.beh.iface + 1;
        TL.beh.face(TL.beh.iface).face_msg = msg;
        TL.beh.face(TL.beh.iface).pup = pup;
        TL.beh.face(TL.beh.iface).time = now;
        TL.beh.eye_position = pup(3) - pi/2;
    end

    % read treadmill information
    TL.beh.irun = TL.beh.irun + 1;
    data = read_treadmill(TL.beh.s_port);
    TL.beh.run_speed(TL.beh.irun)  =   data(3);
    TL.beh.run_time(TL.beh.irun)  =   data(1);
    TL.beh.curr_dist(TL.beh.irun)  =   data(2)-TL.beh.last_dist;
    TL.beh.run_now(TL.beh.irun) = now;
    TL.beh.last_dist = data(2);    

       
    % if more time has passed, go to next stimulus
    ii = TL.stim.ind_stim;
    allow_new_stim = 1;
    if ii>0
        if TL.stim.nflip>=TL.stim.stim_flips(ii)
            if ~isempty(TL.stim.frame)
                Screen('Close', TL.stim.frame)
                TL.stim.frame = [];
            end
        end        
        allow_new_stim = TL.stim.nflip >= TL.stim.stim_flips(ii) + ...
                                TL.stim.grey_flips(ii);
    end
    
    t_microscope = (now - TL.daq.last_mic) * (24 * 3600); 
     if (t_microscope > 1.5/60) && (TL.stim.cond_microscope(TL.stim.ind_stim+1) ==1)
         allow_new_stim = 0;
     end

     if (TL.beh.run_speed(TL.beh.irun)<TL.beh.run_th) && (TL.stim.cond_run ==1)
         allow_new_stim = 0;
     end

    if allow_new_stim
%         disp(TL.stim.cond_microscope(TL.stim.ind_stim+1))
        % new stimulus
        TL.stim.ind_stim = TL.stim.ind_stim + 1;
        % check presentation frequency (should be 7.5Hz)
        % if mod(TL.stim.ind_stim, 75) == 0
        %     disp((now - TL.daq.last_mic) * (24 * 3600));
        % end
        
        % compute new frame
        istim = TL.stim.istim(TL.stim.ind_stim);
        if istim<.5
            frame = .5 ;
        elseif rem(istim,1)>1e-4
            i1 = TL.stim.img(:,:,floor(istim));
            i2 = TL.stim.img(:,:,floor(istim)+1);
            
            Ly = size(i1,2);
            n0 = ceil(rem(istim, 1) * Ly/2);
            frame = cat(2, i1(:, 1:Ly/2), i2, i1(:, Ly/2+1:Ly));  

            frame = cat(2, frame(:, 1+n0:n0+Ly/2), frame(:, 3*Ly/2-n0+1:2*Ly-n0));
%             frame = frame(:, 1+n0:n0+Ly/2);
%             frame = cat(2, frame, fliplr(frame));                        
        else
            frame = TL.stim.img(:,:,floor(istim));
        end
        
        TL.stim.frame = Screen('MakeTexture', TL.stim.window, frame);

        TL.stim.nflip = 0;
    end

    % correct by the eye position
    TL.stim.dshift([1, 3]) = TL.stim.destRect([1,3]) + ...
        TL.beh.eye_position * 2048/(pi/2); %this assume 2048 pixels per 90 degrees

    % need to add an eye movement correction condition here
    delta = 0;
    if exist('dpro') && isfield(dpro, 'dmove') && TL.stim.ind_stim>0
        delta = 0;
        ii = TL.stim.ind_stim;

        cond1 = TL.stim.tmove(ii)>=0;
        cond2 = TL.stim.nflip >= TL.stim.tmove(ii);
        if cond1 && cond2
            delta = TL.stim.dmove(ii);
        end            
           
        % this is additional to whatever the eye correction was
        TL.stim.dshift([1, 3]) = TL.stim.dshift([1,3]) + ...
            delta * 2048/(pi/2); %this assume 2048 pixels per 90 degrees

        if 0 %TL.stim.nflip == TL.stim.tmove(ii)
            disp([TL.stim.nflip, delta])
        end        
    end

    

    if isempty(TL.stim.frame)
        Screen('FillRect', TL.stim.window, .5* TL.stim.colormask);
    else
        Screen('DrawTexture', TL.stim.window, TL.stim.frame, [], TL.stim.dshift, 0,...
            filtermode, [], TL.stim.colormask);
    end


    Screen('AsyncFlipBegin', TL.stim.window, 0, 0);
end

if TL.imaging
	fwrite(u, 'instruction ExpEnd');
    fclose(u);
end

pause(1);

% stop everything else
sca; 
stop(TL.sess.sread);

fpath = fullfile('D:\EXP', mname, datexp, sprintf('%s', blockexp));
write(TL.myMQTT, 'trigger', fullfile(fpath, fname));
unsubscribe(TL.myMQTT, Topic = 'pupil/online');
%unsubscribe(TL.myMQTT,Topic = 'sphericalTreadmill/Data');
fprintf(TL.beh.s_port, 'STOP');
fclose(TL.beh.s_port);

TL.beh.run_speed = TL.beh.run_speed(1:TL.beh.irun);
TL.beh.run_now = TL.beh.run_now(1:TL.beh.irun);

pause(1)

cleanUP23

% cleanup things
% TL.stim.img = []; % too large to save
% if ~isempty(TL.daq.pack)
%     TL.daq.data = [TL.daq.pack.data];
% end
end
%%

function [rpy, n_msg] = read_ball(n_msg, mysub)

rpy = zeros(1,3);
while n_msg<mysub.MessageCount
    msg = mysub.read;
    m = jsondecode(msg{1});
    rpy = rpy + [m.pitch, m.roll, m.yaw];
    n_msg = n_msg+1;
    % 	if rem(n_msg,30)==0
    % 		fprintf('%d, %s \n', n_msg, msg{1})
    % 	end

    pause(1/10000)
end
end

function [rpy, rpy_all] = parse_ball(mqttMsg)
rpy = zeros(1,3);
rpy_all = zeros(size(mqttMsg,1),3);
for j = 1:size(mqttMsg,1)
    msg = mqttMsg(j,2).Data{1};
    m = jsondecode(msg);
    rpy_all(j, :) = [m.pitch, m.roll, m.yaw];
    rpy = rpy + rpy_all(j, :);    
end
end


function [msg, n_msg_face] = read_face(n_msg_face, pupsub)

msg = [];
if n_msg_face == pupsub.MessageCount
    fprintf('%d, no camera message \n', n_msg_face)
end
while n_msg_face<pupsub.MessageCount
    msg = pupsub.read;
    n_msg_face = n_msg_face + 1;
end
end

function s_port = connect_to_treadmil()
    s_port =  serialport('COM3',115200);
    flush(s_port);
    configureTerminator(s_port,"CR");
    fopen(s_port);     
    fprintf(s_port, 'SEND'); 
    flush(s_port);    
end

function data = read_treadmill(s_port)
    flush(s_port);
    fprintf(s_port, 'SEND'); 
    status = strtrim(fscanf(s_port));
    data = sscanf(status,'%f,',[1 4]);
end
    