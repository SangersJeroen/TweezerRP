function GUI6_MDTIF
     % This function starts up the GUI and initializes all it's components.
     % This script also contains the functions so the user can select the
     % necessary Tif¡file. The main function and Play function are also run
     % from here.
     
     %NOTE: if you do not see a screen with 'Enter Data Location', 'start',
     %and 'browse' ... change the Positions in fileloctextbox,
     %filelocbutton, or startbutton (line 14,16,18)
     clear all; close all; clc;
     screensize = get(0,'screensize');
     f = figure('Visible','on','Position',[0,0,screensize(3),screensize(4)],'MenuBar','none','Toolbar','none','Tag','overall');
     % fileloctextbox is a uicontrol object
     fileloctextbox = uicontrol('Style','edit','Callback',@fileloctxt,...
        'String','Enter Data Location','Position',[50 200 450 30],'HorizontalAlignment','left','Tag','fileloctextbox');
     filelocbutton = uicontrol('Style','pushbutton','Callback',@filelocbut,...
         'String',' Browse','Position',[500 200 100 30], 'FontSize',14, 'Tag', 'filelocbutton');
     startbutton   = uicontrol('Style','pushbutton','Callback',@startbut,  ...
         'String','START', 'Position',[350 50 600 150],'FontSize',100,'Tag','startbutton');
     ax1 = axes('Units','pixels','Position',[50 screensize(4)-270 600 200],'Tag','ax1','NextPlot', 'add');
        title(ax1,'X_Position')
     ax2 = axes('Units','pixels','Position',[50 screensize(4)-550 600 200],'Tag','ax2','NextPlot', 'add');
        title(ax2,'Y_Position')
     ax3 = axes('Units','pixels','Position',[710 screensize(4)-550 480 480],'Tag','ax3','NextPlot', 'add');
        title(ax3,'XY_Position')
     progress = uicontrol('Style','text','Visible','off','Position',[50 200 200 30],...
         'String','progress','HorizontalAlignment','left','FontSize',15, 'Tag','progress');
     info = uicontrol('Style','text','Position',[1200 screensize(4)-290 200 200],...
        'String',{'X-stiffness = ?? pN/nm','Y-stiffness = ?? pN/nm'},...%{'Total frames = 0','FPS = 0','Power = 0 mW','X-stiffness = 0 pN/nm','Y-stiffness = 0 pN/nm'},...
        'HorizontalAlignment','left','FontSize',15,'Tag','info');
     playback = uicontrol('Style','pushbutton','Callback',@playbut,...
        'String','PLAY','Position',[1200 screensize(4)-560 200 50],'FontSize',20,'Tag', 'playback');
     set(f,'Name','Optical Tweezer analysis','numbertitle','off')
     
%      handles.fileloctextbox = findobj(f, 'Tag', 'fileloctextbox');
%      handles.filelocbutton = findobj(f, 'Tag', 'filelocbutton');
%      handles.startbutton = findobj(f, 'Tag', 'startbutton');
%      handles.progress = findobj(f, 'Tag', 'progress');
%      handles.info = findobj(f, 'Tag', 'info');
%      handles.playback = findobj(f, 'Tag', 'playback');
     
     filename = 'empty';
     fileloc = 'empty';

    function  fileloctxt(source,eventdata, handles)
    % % % %      fileloc = get(handles.fileloctextbox,'String')
         hh = findobj(f, 'Tag', 'fileloctextbox');
         fileloc = get(hh,'String')
         [fileloc,filename,ext] = fileparts(fileloc);
         filename = [filename ext];
    end

     function  [fileloc,filename,ext]=filelocbut(source,eventdata,handles)
         [filename fileloc] = uigetfile({'*.tif','TIFF files(*.tif)';'*.avi','AVI files(*.avi)'});
         AA=[fileloc, filename]; 
         hh = findobj(f, 'Tag', 'fileloctextbox');
         set(hh,'String',AA);
    %      fileloc = fileloc(1:end-1)
     end

     function startbut(source,eventdata,handles)
         hh = findobj(f, 'Tag', 'startbutton');
         set(hh,'String','Calculating');
    % %      set(handles.startbutton,'String','Calculating...')
         drawnow()
         skip_fpf=0;
         l_pixel=(5.2e-6/60/.5);
         [X,Y,xstiff,ystiff,num,pow,fps] = Main_MD(skip_fpf,progress,l_pixel,f);
         set(hh,'String','START');
         ax1 = findobj(f, 'Tag', 'ax1');
         size(X)
         cla(ax1),
         plot(ax1,X)
         title(ax1,'X¡Position')
         xlabel(ax1,'frame number')
         ylabel(ax1,'pixel position')
         grid(ax1,'ON')

         ax2 = findobj(f, 'Tag', 'ax2');
         cla(ax2),
         plot(ax2,Y)
         title(ax2,'Y¡Position')
         xlabel(ax2,'frame number')
         ylabel(ax2,'pixel position')
         grid(ax2,'ON')

         ax3 = findobj(f, 'Tag', 'ax3') % if I don't do this after 1x start, 1x play, the Tag is lost with yet another play
         set(ax3,'NextPlot', 'add')
          cla(ax3), 
         plot(ax3,X,Y,'x')
         ax3 = findobj(f, 'Tag', 'ax3')
         title(ax3,'XY-Position')
         xlabel(ax3,'X-position (pixels)')
         ylabel(ax3,'Y-position (pixels)')
         grid(ax3,'ON')
    
        AA={ ...%['Total frames = ' num2str(num)],['FPS = ' num2str(fps)],['Power = ' num2str(pow) ' mW'],...
             ['X-stiffness = ' num2str(xstiff) ' pN/nm'],...
             ['Y-stiffness = ' num2str(ystiff) ' pN/nm']}
        hh = findobj(f, 'Tag', 'info');
        set(hh,'String',AA);
     end
      % position=[X,Y]; 
     function playbut(source,eventdata,ax1,ax2,ax3)
         hh = findobj(f, 'Tag', 'fileloctextbox')
         fileloc = get(hh,'String')
         [fileloc,filename,ext] = fileparts(fileloc); 
         fileloc=[fileloc,'\'];
%          set(hh,'String','START');
         ax1 = findobj(f, 'Tag', 'ax1');
         ax2 = findobj(f, 'Tag', 'ax2');
         ax3 = findobj(f, 'Tag', 'ax3')
        
         Play(fileloc,filename,ext,ax1,ax2,ax3);
         ax3 = findobj(f, 'Tag', 'ax3')
    %     end
         % With this part of the code the user can edit the screen and all the
        % components will also alter their lenght and width accordingly.
         playback.Units = 'normalized';
         info.Units = 'normalized';
         progress.Units = 'normalized';
         filelocbutton.Units = 'normalized';
         fileloctextbox.Units = 'normalized';
         ax1.Units = 'normalized';
         ax2.Units = 'normalized';
         ax3.Units = 'normalized';
         startbutton.Units = 'normalized';
    end

    function Play(fileloc,filename,ext,ax1,ax2,ax3)
     % This function is used to play back the data. It also starts of by
     % filtering out if file location is valid. If the file location is
     % valid the script loops trough all the frames of the tiff file and
     % outputs the necessary data to the GUI.
         warning('');
         addpath(fileloc);
         A=dir(fileloc);
         if isequal(lastwarn,['Name is nonexistent or not a directory: ' fileloc])
            warndlg('Please first enter a valid location')
         else
            if not(exist([fileloc,filename,'.mat']))%if not(exist([filename(1:end-4),'.mat']))
                warndlg('Please first start calculating data')
            else
                [fileloc,filename,'.mat']
                 load([fileloc,filename,'.mat'],'Xest','Yest', 'noiseI')% load([filename(1:end-4),'.mat'],'Xest','Yest')
                 num = numel(Xest);
                 cla(ax1),
                 plot1 = plot(ax1,Xest(1));
                 title(ax1,'X¡Position')
                 xlabel(ax1,'frame number')
                 ylabel(ax1,'pixel position')
                 grid(ax1,'ON')

                cla(ax2),
                 plot2 = plot(ax2,Yest(1));
                 title(ax2,'Y¡Position')
                 xlabel(ax2,'frame number')
                 ylabel(ax2,'pixel position')
                 grid(ax2,'ON')

                 [fileloc,filename,ext]
                  if isequal(ext,'.avi')
                    video=VideoReader([fileloc,filename,ext]);
                    num = get(video, 'NumberOfFrames');
                else %tif
                    video=1;
                    info = imfinfo([fileloc,filename,ext]);
                    num = numel(info);
                  end
                 

                 AA=Import(1,video,[fileloc,filename,ext]);
%                  whos AA
                 cla(ax3),
                
                 plot3=image(AA,'Parent',ax3) ;
                 hold(ax3,'on')
                 plot4 = plot(ax3,Xest(1),Yest(1),'xw');
                 hold(ax3,'off')
                 title(ax3,'XY¡Position')
                 xlabel(ax3,'X¡position (pixels)')
                 ylabel(ax3,'Y¡position (pixels)')

                 set(ax1,'Xlim',[1 num])
                 set(ax2,'Xlim',[1 num])
                for i = 1:num
                     set(plot1,'Ydata',Xest(1:i))
                     set(plot2,'Ydata',Yest(1:i))
%                      set(plot3,'Cdata',0.3*Import(i,video)) % it would be nice to change the caxis scaling. 
                     set(plot3,'Cdata',0.3*noiseremover(Import(i,video,[fileloc,filename,ext]),noiseI)) % Also wouldn't it be better to look at image after noiseremover
                     set(plot4,'Xdata',Xest(i))
                     set(plot4,'Ydata',Yest(i))
                     drawnow()

                end
            end
        end
    end

    function [X,Y,xstiff,ystiff,xst,yst,num,pow,fps] = Main_MD(skip_fpf,progress,l_pixel,f)
     % From this function all the other scripts are run. It starts of by
     % filtering out a warning which is given if the file location isn't
     % correct. If this isnt the case the main code starts of. This code
     % checks if there is already a save¡file containing the date. If this
     % is not the case the data is calculated. Otherwise the save file is
     % loaded. The X¡ and Y¡position are then used to calculate the trap
     % stiffness. The variance can be edited to make the quadrant tracker
     % even more accurate, but this will greatly increase calculation time.

     % MD 180322: adapted the Main PEP m-file to accept avi files instead of tiff files
        hh = findobj(f, 'Tag', 'fileloctextbox');
        fileloc = get(hh,'String')
        [fileloc,filename,ext] = fileparts(fileloc);
        fileloc=[fileloc,'\'];
%          disp(fileloc);
%          disp(filename);
         mindx = 0.01;

         warning('');
         addpath(fileloc);
         A=dir(fileloc);

         num = 0;
         xstiff = 0;
         ystiff = 0;
         pow = 0;
         fps = 0;
         %l_pixel=46.8e-9; %meter this is in D1.210 setup with 100x oil objective, 
         %l_pixel=86.7e-9;     % G0.380 setup where we have 60x water objective, still to be verified
         X = [];
         Y = [];
         if isequal(lastwarn,['Name is nonexistent or not a directory: ' fileloc])
            warndlg('Please first enter a valid location')
         elseif isequal(ext,'.avi') ||isequal(ext,'.tif')
             if isequal(ext,'.avi')
                video=VideoReader([fileloc,filename,ext]);
                num = get(video, 'NumberOfFrames');
             else %tif
                 video=1;
                 info = imfinfo([fileloc,filename,ext]);
                 num = numel(info);
             end
            Xest = zeros(1,num);
            Yest = zeros(1,num);
            [fileloc,filename,'.mat']
            exist([fileloc,filename,'.mat'])
            if not(exist([fileloc,filename,'.mat']))% not(exist([filename(1:end-4),'.mat']))
                I = Import(1,video, [fileloc,filename,ext]);
                [h,b] = size(I);
                noiseI = noisecalculator(I);
%                 I = noiseremover(I,noiseI);
                [xCOM,yCOM] = centerofmass(I); % previously COM done for each frame
                QI=TrackXY_by_QI_Init(I);
                [XQI,YQI]=TrackXY_by_QI(single(I),QI,0,xCOM,yCOM);
    %                   [XQI,YQI]=TrackXY_by_QI(single(I>0),QI,0,xCOM,yCOM);
                Xest(1)=XQI;
                Yest(1)=YQI;
                hh = findobj(f, 'Tag', 'progress');
                set(hh,'String',['Progress: 0 / ' num2str(num)])
                set(hh,'Visible','on')
                drawnow()
                for j = 2:num
                    clear I xest yest dx dy h b
                    if mod(j,10) == 0
                             set(hh,'String',['Progress: ' num2str(j) ' / ' num2str(num)])
                             drawnow()
                    end
                    I = Import(j,video,[fileloc,filename,ext]);% % if isequal(filename(end-2:end),'tif'), I = Import(filename,j);
%                     I = noiseremover(I,noiseI);
                    [Xest(j),Yest(j)]=TrackXY_by_QI(single(I),QI,0,XQI,YQI);
                end
                save([fileloc,filename,'.mat'],'Xest*','Yest*', 'noiseI')
                set(hh,'Visible','off')
                drawnow()
            else
                load([fileloc,filename,'.mat'],'Xest','Yest','noiseI')
            end
         end
         xst = stiffnes(Xest,l_pixel);
         xstiff = xst;%spectrap(Xest,xst,l_pixel);
         yst = stiffnes(Yest,l_pixel);
         ystiff = yst;%spectrap(Yest,yst,l_pixel);
         if not(isempty(strfind(filename,'fps')))
                fps = str2num(filename((strfind(filename,'_')+1):(strfind(filename,'fps')-1)));
         else
                fps = 100;
         end
         pow = str2num(filename(2:5))/100;
         X = Xest;
         Y = Yest;
    end
end
%% import function
%Approximation of the trap efficiency of an Optical Tweezer. Written by M.W. Docter in 2017
%for the PEP2 group. Propaedeutic End Project (PEP) is a part of the bachelor
 %program Applied Physics and Applied Mathematics at the Technical University of Delft.

% clear all
% close all
% 
% sigm=1:750;
% trap_eff=zeros(1,length(sigm))
% for ss=1:length(sigm)
%     phase=-360:360; %Angle
%     gf=exp(-(phase/sigm(ss)).^2); %Gauss function
%     gf=gf/sum(gf(:)); %Scaling Gauss Function
%     ha=1-(cosd(phase)).^2; % Angle Dependency
%     ha(abs(phase)>90)=0; % Only calculate between -90 and +90 degrees
%     figure(1),
%     subplot(2,1,1),plot(phase,gf,'r',phase,ha,'b')
%     subplot(2,1,2),plot(phase,gf.*ha)
%     trap_eff(ss)=sum(gf.*ha); % integral= sum Gauss function * Angle Dependency
% end
% figure(3), plot(sigm, trap_eff)
% xlabel('Beam thickness [ m ]')
% ylabel('Beam Intensity [J]')

%%
function [output] = Import(num,video,loc)
%%function [output] = Import(num,video)
 % This function is created to import pictures, including individual
 % pictures from .tif¡files. To read a whole .tif¡file, a for¡loop could be
 % used.
 if isequal(loc(end-2:end),'tif')
     tmp = imread(loc,num); % imports from location x the y¡th image.
     tmp = tmp(2:end-1,:); % deletes most upper and lower pixel rows, since they contain no data.
 else
    tmp = read(video,num);
 end
    I=(tmp(:,:,1));
    I(isnan(I)) = 0; % sets 'Not a Number'¡values to 0
    I = single(I); % sets data type of Im from uint8 to single, for further calculations
    if any(I(:) <0)
        warning('Image contains negative values');
    end
    output = I;
end
%%
function [ noiseI ] = noisecalculator( I )
 %This function calculates the amount of noise in an image. The returned
 %value 'noiseI' is the noise intensity which should be removed from all
 %images in a .tif¡file.
 % It's probably sufficient to run this function once for every .tif¡file,
 % since approximately all images in the file contain the same amount of
 % noice. This will reduce the calculation time drastically.

    %% Delete background intensity
    med = median(I(:)); % median of all intensities equals background intensity
    I = abs( I - med );

    %% Calculate the noise intensity
    ilst = sort( I(:) ); % create sorted list of all present intensities
    b = numel(ilst);
    x = 1:b;
    ilst = ilst'; % transpose
    coeff = polyfit( x(floor(b/2):b ), ilst(floor(b/2):b ), 1); % makes a linear fit to the highest half of all intensities in 'ilist'. 
    % The rightmost intersection is the chosen noise intensity which will be removed from all images.
    fit = polyval(coeff,x); % returns an array of all values of the fitted line for every x¡coordinate
    dif = (ilst-fit); % calculates the difference between the true intensities and the values of the fitted line
    ruisIx = find( diff( sign( dif )), 1, 'last') + 1; % calculates the x¡value of the rightmost intersection of the fitted line and the curve of all true intensities
    noiseI = ilst(ruisIx); % calculates the intensity, which belongs to the above¡calculated x¡value. This equals the intensity of the noise which should be delete.
end
%%
function [ A ] = noiseremover( I, noiseI )
 % This function deletes the background intensity and noise.
 % For determining the background intensity, the median of all
 % intensities is calculated, and is then substracted from every
 % pixel intensity. After that, the noise intensity 'noiseI' is
 % substracted from every pixel intensity.

 %% DELETE NOISE
med = median(I(:)); % median of all intensities equals background intensity
Ib = abs( I - med ); % delete background intensity from image
Ic = Ib - noiseI; % delete noise intensity from image
Id=Ic;
Id(Id<0) = 0; % delete all negative intensities, so only the high intensities which contain
 % useful data are left, all noise and background colors should now be gone
A = I;
end
%%
function [X, Y] = centerofmass(I)
% Calculation of the COM. The X and Y coordinates of the COM are
% returned.

    %% VARIABLES
    ndim = 2; % since we only use greyscale images, every image is a 2dimensional array in matlab

    %% CALCULATION OF THE CENTER OF MASS (COM)
    sz = size(I); % size of I
    M = sum(sum(I)); % summation of all pixel intensities
    coord = zeros(1,ndim); % matrix which will be filled with the COM¡coordinates
    if M==0 % if completely black image, then empty array is returned
         X = 0;
         Y = 0;
    else
         for dim = 1:ndim
         a = ones(1,ndim);
         a(dim) = sz(dim);
         r = sz;
         r(dim) = 1;
         ind = repmat(reshape(1:sz(dim),a),r); % creates matrix filled with position values in current dimension
         coord(dim) = sum(sum(ind.*I))./M; % calculation of COM in 1 dimension
         end
        Y = coord(1); % returns COM coordinates
        X = coord(2);
    end
end
%%

% function [output] = Radius(h,b,x,y)
%     r = [x y h-y b-x];
% 	R = floor(0.95*min(r));
% 	output = R;
% end
%%
function [output] = polar(I,xcent,ycent,R)
 % This function is used tot calculate a polar profile from an intensity image I.
 % The coordinates xcent and ycent are used as center. The r is the maximum radius
 % for the polar profile.
 % According to Loenhout, 2012, the optimal result is gained when the
 % value dr is one third of a pixel length.
 % For d(/theta) we use one degree.

    Ip = zeros(360,3*R);
    [h, b] = size(I);
    % Empty matrix to fill with the intensity's in polar cordinates. The y
    % coordinates resemble the angle and the x coordinates the radius.
    for a=1:360 % when a == 0, we are on the y¡axis. When a increases, we turn clockwise
        for r=1:3*R
             xcord=xcent+r/3*sind(a); % Calculating Cartesian cordinates of position r,a in the matrix.
             ycord=ycent+r/3*cosd(a);
             xlow=floor(xcord); % Calculating the four coordinates in the Intensity matrix that are around xcord,ycord
             xhigh=xlow+1;
             ylow=floor(ycord);
             yhigh=ylow+1;
             xfactor=mod(xcord,1);
             yfactor=mod(ycord,1);
            if xlow>0 && ylow>0 && yhigh < h && xhigh < b % The program crashes when x or y are below 1 or above the image radius
                % Interpolating over first x then y
                Ip(a,r)=(1-yfactor)*((1-xfactor)*I(ylow,xlow)+xfactor*I(ylow,xhigh))+yfactor*((1-xfactor)*I(yhigh,xlow)+xfactor*I(yhigh,xhigh));
            end
        end
    end
    output = Ip;
end
%%
function [Xxx, Yyy] = qi( Ip, R )
 %Quadrant interpolation as described in Loenhout, 2012
 % Xxx and Yyy arrays are the cross correlations in x and y directions. To
 % determine the dx and dy the maximum of this array should be calculated
 % as follows:
 % dx = (2/(3*pi))*(max(Xxx) ¡ 2*R);
 % dy = (2/(3*pi))*(max(Yyy) ¡ 2*R);
 % this is a correction for a factor 2/pi from the xcorrelation and a factor
 % 1/3 from the polar coordinates. This is done in the fivepointfit
 % script, for an optimal
 % Ip is the polar profile from the image, R is the radius of the
 % polar profile.

 % FORCE into column vectors
 qtr=zeros(R,1);
 qbr=zeros(R,1);
 qtl=zeros(R,1);
 qbl=zeros(R,1);
    for i = 1:R
        qtr(i) = mean(Ip(1:90,i)); % intensity profile right top quadrant, using the mean over the quadrant per dr
        qbr(i) = mean(Ip(90:180,i)); % intensity profile right bottom quadrant
        qbl(i) = mean(Ip(180:270,i)); % intensity profile left bottom quadrant
        qtl(i) = mean(Ip(270:360,i)); % intensity profile left top quadrant
    end
	r = (1:R);
	
	qr = qtr + qbr; % intensity profile right half
	med1 = median(qr(:));
	qra = qr - med1; % normalising using median
	
	ql = qbl + qtl; % intensity profile left half
	med2 = median(ql(:));
	qla = ql - med2;
	
	minqla = flipud(qla);
	Ix = [minqla; qra]; % array where the mirror image of the left half and the right half are concatenated
 
	qt = qtr + qtl; % intensity profile top half
	med3 = median(qt(:));
	qta = qt - med3;
 
	qb = qbr + qbl; % intensity profile bottom half
	med4 = median(qb(:));
	qba = qb - med4;
 
	minqba = flipud(qba);
	Iy = [minqba; qta]; % array where the mirror image of the bottom half and the top half are concatenated
 
   % from paper: 
	Xxx = xcorr(Ix, flipud(Ix)); % cross correlation in x direction
	Yyy = xcorr(Iy, flipud(Iy)); % cross correlation in y direction
end
%%
function [ dx, dy ] = fivepointfit( Xxx,Yyy,R )
 % Fivepointfit: The change in position dx of Xxx and dy of Yyy are
 % calculated
 % The x¡coordinate for the max value of Xxx and Yyy are calculated. Secondly,
 % 3 points are chosen around this point, (with steps of 3). A polyfit is
 % used to fit a parabola through these points and finally the x¡value of
 % the peak of this parabola is determined. The x¡value of the peak
 % contains information about the correction for the center of mass.
 % This value should be adopted with a factor 2/(pi) for the
 % cross¡correlation and a factor 1/3 because of the way the polar
 % coordinates are chosen.
 % input are the cross¡correlation arrays and the chosen radius for
 % the polar coordinates, output is the correction for the center
 % determined before.

    x = Xxx;
	ix = find(x == max(Xxx));
	x2 = [ix-6 ix-3 ix ix+3 ix+6];
	y2 = [x(ix-6) x(ix-3) x(ix) x(ix+3) x(ix+6)];
	P = polyfit(x2,y2,2);
	dxa = -P(2)/(2*P(1));

    y = Yyy;
	ix2 = find(y == max(Yyy));
	x3 = [ix2-6 ix2-3 ix2 ix2+3 ix2+6];
	y3 = [y(ix2-6) y(ix2-3) y(ix2) y(ix2+3) y(ix2+6)];
	P2 = polyfit(x3,y3,2);
	dya = -P2(2)/(2*P2(1));
 
  % Note that the values for dxa and dya are actually in polar coordinates.The dxa and dxy are
  % translated to the center of the beat (so dxa and dxy are the differences with the
  % previous determined center point). After this, a factor 1/(3*pi) is used
  % in order to get the dx and dy in cartesian coordinates.
 
	dx = (2/(3*pi))*(dxa - 2*R);
	dy = (2/(3*pi))*(dya - 2*R);
end
%%
function [ k ] = stiffnes(Xest, l_pixel)
 %input is a list of x¡ and y¡locations (in pixel) per time of a bead in a non¡moving
 %trap with no other forces acting on it, and the temperature (and kb)
     Pos = [Xest]; %length in pixel
     T=293.15; %Kelvin
     kb=1.38064852e-23; %Joule/Kelvin
     
     %conversion to actual position
     x=Pos*l_pixel; %metre
     kb=1.38064852e-23; %Joule/Kelvin
     %calculation of average x
     x_mean=mean(x(:));
     dx=x-x_mean;
     %calculation of k
     k = kb*T./(mean(dx(:).^2))*10^3; %pN/nm
end
%%
function [Ktr] = spectrap(XYest, kxyest, l_pixel)
 %SPECTRAP Fit for the trap stiffness Ktr
 % Using power spectrum analysis
 %% initiating parameters
     sample = 150; %Hertz
     yg = 6*pi*10^-3*10^-6;%drag coefficient
     T=293.15; %Kelvin
     kb=1.38064852e-23; %Boltzmann constant
     D = kb*T/yg;% diffusion coefficient
     

     %%fit
     Pos = XYest*l_pixel;
     fsq = (kxyest/(2*pi*yg))^2;
     ydata = abs(fftshift(fft(Pos))); %input position of center
     N = numel(ydata);
     xdata = linspace(-sample/2,sample/2,N+1); %frequency
     xdata=xdata(1:end-1);
     xx = find(xdata==0);
     ydata(xx) = ydata(xx+1);% deleting giant peak
     xdata = xdata(xx+1:end);
     ydata = ydata(xx+1:end);
    csydata = cumsum(ydata);
    fun = @(x,xdata) cumsum(x(1)*(xdata.^2+x(2)).^-1); %creating function for lsqcurvefit combined with cumsum
    x0=[D/(pi^2),fsq]; %estimate parameters
     options = optimoptions(@lsqcurvefit,'Display','Iter');
     x = lsqcurvefit(fun,x0,xdata,csydata);
     f0 = sqrt(x(2));
     Ktr = f0*(2*pi*yg); % fitted ktrap value
end

function  pretracksettings=TrackXY_by_QI_Init(firstim)
%JWJK:This function intializes settings for a sub-pixel XY fit by making 4 profiles in QI
%style. Algorithm described in 
%[1]: M.T.J. van Loenhout, J. Kerssemakers , I. De Vlaminck, C. Dekker
% Non-bias-limited tracking of spherical particles, enabling nanometer 
%resolution at low magnification
% Biophys. J. 102, Issue 10, 2362 (2012)

%Set includes following functions:
% TrackXY_by_QI: main
% TrackXY_by_QI_Init: initialization via first image 
% TrackXY_by_COM_2Dmoment: first gues center-of mass
% SymCenter: sub-pixel 1D fit
% MakeHighResRing: genarates artificial image for demo purposes

%Jacob Kerssemakers, 2017
%:JWJK--------------------------------------------

%general settings, see ref [1] for details
    pretracksettings.radialoversampling=2;  
        %default 2 (per image pixel length)
    pretracksettings.angularoversampling=0.7; 
        %deault 0.7 typically covers all image pixels
    pretracksettings.minradius=0; 
    pretracksettings.maxradius=min(size(firstim))/2.5;
    pretracksettings.iterations=10; 
        %number of times refinement of center is repeated; typically minimal 5        
    pretracksettings=Build_QI_SamplinggridGrid(pretracksettings) ;
        %relative coordinates of radial sampling grid; calculatedbeforehand
        %to save time
end  

function QI=Build_QI_SamplinggridGrid(QI) 
    %Build a radial sampling grid; based on image size
    spokesnoperquad=ceil(2*pi*QI.maxradius*QI.angularoversampling/4);
    radbinsno=(QI.maxradius-QI.minradius)*QI.radialoversampling;
    angles=linspace(-pi/4,2*pi-pi/4,spokesnoperquad*4+1)'; 
    angularstep=pi/2/spokesnoperquad;
    angles=angles(1:end-1)+angularstep/2; %to center angles per quadrant
    radbins=linspace(QI.minradius,QI.maxradius,radbinsno);
    [argsgrid,radiigrid]=meshgrid(angles,radbins);
    QI.Y0samplinggrid=(radiigrid.*sin(argsgrid))';
    QI.X0samplinggrid=(radiigrid.*cos(argsgrid))';
    QI.angles=angles;
    QI.radbii=radbins;
end

function [xnw,ynw]=TrackXY_by_QI(im,QI,sho,xm,ym)
%JWJK:
%------------------------------------------------------------
%This function prepares a sub-pixel XY fit by making 4 profiles in QI
%style. Algorithm described in 
% M.T.J. van Loenhout, J. Kerssemakers , I. De Vlaminck, C. Dekker
% Non-bias-limited tracking of spherical particles, enabling nanometer resolution at low magnification
% Biophys. J. 102, Issue 10, 2362 (2012)

%Set includes following functions:
% TrackXY_by_QI: main
% TrackXY_by_QI_Init: initialization via first image 
% TrackXY_by_COM_2Dmoment: first gues center-of mass
% SymCenter: sub-pixel 1D fit
% MakeHighResRing: genarates artificial image for demo purposes

%To run it: 
   %run TrackXY_by_QI from command line; no input runs auto-generated image demo - try it! 
   %in script: use first 'TrackXY_by_QI_Init' on a first image;
   %then use 'TrackXY_by_QI';

%input:    
    %im: input image; should contain reasonably cicrcle-symmetric pattern.
    % QI: structure containing presets for tracking; see function 'TrackXY_by_QI_Init'  
    %sho: set to 1 to se demo output, otherwise zero
%output:
    %xnw,ynw: absolute x and y coordinates of pattern center, image pixel units

 %Matlab Version written by Jacob Kerssemakers. Disclaimer: original code
 %was tested and published in Labview; this Matlab Code was not rigorously
 %tested; bugs may be present. User is expected to understand Matlab.
 %Jacob Kerssemakers, 2017
 %:JWJK---------------------------------------------------------------------

    if nargin <3
        sho=1;
        close all
        test.PicSize=50; 
        x0=test.PicSize/2+17; 
        y0=test.PicSize/2; 
        test.PatternRingRadius=test.PicSize/8;
        test.PatternRingWidth=test.PicSize/30;
        im=MakeHighResRing(x0,y0,test); 

        QI.radialoversampling=2;
        QI.angularoversampling=0.7;
        QI.minradius=0;
        QI.maxradius=test.PicSize/3;
        QI.iterations=10;
        QI=TrackXY_by_QI_Init(im);
     end  
     %close all   

    %iterative section
    % [xm,ym]=TrackXY_by_COM_2Dmoment(im); %center-of-mass for initial guess
     xnw=xm;
     ynw=ym;

    errx=zeros(QI.iterations,1);
    prequit=0;
    %iterative section; center position is iteratively improved
    for ii=1:QI.iterations   
         if ~prequit %Note that this may stop the loop prematurely 
            xol=xnw;
            yol=ynw;
            Xsamplinggrid=QI.X0samplinggrid+xnw;
            Ysamplinggrid=QI.Y0samplinggrid+ynw;
            allprofiles=(interp2(im,Xsamplinggrid,Ysamplinggrid,'NaN'));

            [aa,rara]=size(Xsamplinggrid);
            spokesnoperquad=round(aa/4);
            Qiprofs=zeros(4,rara);
            Qiprofs(1,:)=nanmean(allprofiles(1:spokesnoperquad,:));  %East
            Qiprofs(2,:)=nanmean(allprofiles(spokesnoperquad+1:2*spokesnoperquad,:));  %North
            Qiprofs(3,:)=nanmean(allprofiles(2*spokesnoperquad+1:3*spokesnoperquad,:));  %West
            Qiprofs(4,:)=nanmean(allprofiles(3*spokesnoperquad+1:4*spokesnoperquad,:));  %South

            QiHor=[fliplr(Qiprofs(3,:)) Qiprofs(1,:)];
            QiVer=[fliplr(Qiprofs(4,:)) Qiprofs(2,:)];

            %Get-image centered position, corrected for oversampling and off-center
            %sampling
            fudgefactor=(pi/2);  %see ref [1]
            xnw=-((length(QiHor)/2-SymCenter(QiHor))+0.5)/QI.radialoversampling/fudgefactor+xnw;
            ynw=-((length(QiVer)/2-SymCenter(QiVer))+0.5)/QI.radialoversampling/fudgefactor+ynw;
            errx(ii)=((xnw-xol)^2+(ynw-yol)^2);
            if (isnan(xnw)||isnan(ynw)) 
                prequit=1;
            end  
         else
             prequit=1;
             xnw=xol;  %fetch the old values
             ynw=yol;
         end        
    end
    disp([xol,xnw])
    if (isnan(xnw)||isnan(ynw)) 
            xnw=xol;  %fetch the old values
             ynw=yol;
    end 
    if nargin <3  %Plotting Menu    
         [aa,~]=size(Xsamplinggrid);
         spokesnoperquad=round(aa/4);          
         subplot(2,2,1);
             hold off
             pcolor(im); colormap bone; shading flat; hold on; 
             plot(Xsamplinggrid(1:spokesnoperquad,:)+0.5,Ysamplinggrid(1:spokesnoperquad,:)+0.5,'r-');
             plot(Xsamplinggrid(spokesnoperquad+1:2*spokesnoperquad,:)+0.5,Ysamplinggrid(spokesnoperquad+1:2*spokesnoperquad,:)+0.5,'k-');
             plot(Xsamplinggrid(2*spokesnoperquad+1:3*spokesnoperquad,:)+0.5,Ysamplinggrid(2*spokesnoperquad+1:3*spokesnoperquad,:)+0.5,'r-');
             plot(Xsamplinggrid(3*spokesnoperquad+1:4*spokesnoperquad,:)+0.5,Ysamplinggrid(3*spokesnoperquad+1:4*spokesnoperquad,:)+0.5,'k-');
             plot(xm+0.5,ym+0.5,'wx', 'Markersize', 15);
             plot(xol+0.5,yol+0.5,'w+', 'Markersize', 15);
             plot(xnw+0.5,ynw+0.5,'wsq', 'Markersize', 15);     
             title('Image&sampling grid') ;          
         subplot(2,2,2);
             hold off
             plot(Qiprofs'); hold on;
             title('profiles');
             xlabel('sampling steps');
             ylabel('value, a.u.');
             legend('East', 'North' , 'West', 'South');     
         subplot(2,2,3);
             hold off
             plot(QiHor, 'r-'); hold on;
             plot(QiVer);
             title('concatenated profiles')
             legend('Hor', 'Ver');
             xlabel('sampling steps');
             ylabel('value, a.u.');          
         subplot(2,2,4);
             hold off
             plot(errx,'k-o'); hold on;
             title('iteration progress')
             legend('Hor', 'Ver');
             xlabel('iteration step no.');
             ylabel('translation per step, pixel units'); 
             pause(0.1);
    end
end

function x=SymCenter(prf);
    %this function find the symmetry center of an array.  
    %Jacob Kerssemakers-----------------------------------
    mp=nanmean(prf);
    sel=find(isnan(prf)); prf(sel)=mp;  %padding nans
    fw=prf-nanmean(prf);             %forward
    rv=fliplr(prf)-nanmean(prf);     %reverse
    d=real(ifft(fft(fw).*conj(fft(rv))));
    ld=ceil(length(d)/2);
    d=[d(ld+1:length(d)) d(1:ld)]';   %swap first and second half 
    [val,x]=max(d);
    x=(subpix_step(d)+length(prf)/2)/2;    
end

 function x=subpix_step(d);
    %this function performs a subpixel step by parabolic fitting
    hf=3; ld=length(d);  xs=[1:1:ld]';   [~,x]=max(d);  %uneven hf
    lo=max([x-hf 1]); hi=min([x+hf ld]);  %cropping
    ys=d(lo:hi); xs=xs(lo:hi);
    prms=polyfit(xs,ys,2); x=-prms(2)/(2*prms(1));
 end
 