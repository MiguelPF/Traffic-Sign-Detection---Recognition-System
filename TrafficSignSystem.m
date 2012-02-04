function TrafficSignSystem()
%% Complete Traffic Sign detector and recognition system
tic
clear all;
aviobj = avifile('video_KUL.avi','FPS',8,'compression','Indeo5');
directory = '.\KUL-seq01';        %Images folder
% directory = '.\CVC-02-seq-05';
files = ListFiles(directory);
images_dir = '.\ModelsRecognition';
contador=0;
images_file = ListFiles(images_dir);
load model_HOG_432.mat;  % Carga lo SVs resultantes del entreno
for i=1:size(files,1)
    disp(i);
    % Read file
    im = imread(strcat(directory,'/',files(i).name));
    % -- Auto White Balance Algorithm
    %     im = wbalance(im);
    [segmentedIm]=SegmentIm(im);
    %     nombreruta=strcat('C:\Documents and Settings\Miguel\Escritorio\CV\week11\segmentedIm_KUL','/',files(i).name,'.mat');
    %     save(nombreruta, 'segmentedIm');
    %     load(nombreruta);
    % Candidate Generation (window)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Geberamos las ventanas que contienen areas de pixeles validos
    
    windowCandidates = CandidateGenerationWindow(segmentedIm);
    Detection_Windows=detectorBB_Sliding_SVM(windowCandidates,im,model_HOG); %Detector completo (Sliding+Clasificador+Clustering)    
    [Predicciones,error]=trafficSignRecognizer(im,Detection_Windows);
    
    % Dibujamos la imagen de la señal detectada encima de la original
    if ((isempty(Predicciones)==0)&(Predicciones~=0))
    for l=1:length(Predicciones)
        if (error(l)<=0.75)
        signal = imread(strcat(images_dir,'/',images_file(Predicciones(l)).name));
        signal_res = size(signal);
        im(1:signal_res(1),1+100*(l-1):signal_res(2)+100*(l-1),:) = signal;
        end
    end
    end
    
    figure(1),
    subplot(1,2,1), subimage(im), set(gca,'xtick',[],'ytick',[]),
    if (isempty(windowCandidates)==0)
        for l = 1:numel(windowCandidates)
            figure(1),rectangle('Position',[windowCandidates(l).x windowCandidates(l).y windowCandidates(l).w windowCandidates(l).h],'EdgeColor','r');
        end
    end
    if (isempty(Detection_Windows(1).x)==0)
        for l = 1:numel(Detection_Windows)
            figure(1),rectangle('Position',[Detection_Windows(l).x Detection_Windows(l).y Detection_Windows(l).w Detection_Windows(l).h],'EdgeColor','y');
        end
    end,
    subplot(1,2,2), subimage(segmentedIm.*255), set(gca,'xtick',[],'ytick',[]),
    fig=figure(1);
    F = getframe(fig);
    aviobj = addframe(aviobj,F);
end
close(fig);
aviobj = close(aviobj);
disp('Tiempo de cálculo en segundos'), disp(toc)

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CandidateGeneration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function windowCandidates = filter_Bounding(Bounding,Area)
NumObjects = numel(Area);
j = 1;
MinArea = 480;
MaxArea = 40000;
aspect_ratio = 1.5;
% --
buffer(NumObjects)=struct('x',[],'y',[],'w',[],'h',[]);   %Preallocate
for i=1:NumObjects
    if ((Area(i).Area >= MinArea) && (Area(i).Area <= MaxArea) && (Bounding(i).BoundingBox(3) <= aspect_ratio * Bounding(i).BoundingBox(4)))
        %Filter by area
        buffer(j).x = Bounding(i).BoundingBox(1);
        buffer(j).y = Bounding(i).BoundingBox(2);
        buffer(j).w = Bounding(i).BoundingBox(3);
        buffer(j).h = Bounding(i).BoundingBox(4);
        j = j + 1;
    end
end
windowCandidates(1:j-1)=buffer(1:j-1);
end

function [windowCandidates] = CandidateGenerationWindow(im)
[Bounding,Area]=bounding_boxes(im);
n = numel(Area);
if n>0
    windowCandidates = filter_Bounding(Bounding,Area);
else
    windowCandidates = [];
end
end

function [Bounding,Area] = bounding_boxes(mask)
%First we label each connected region
maskLabeled = bwlabel(mask, 8); %4 or 8 indicates number of sorrounding pixels
%The we calculate the boundix box cointaining each of the regions
Bounding = regionprops(maskLabeled, 'BoundingBox');    %returns a struct with bounding box
Area = regionprops(maskLabeled,'Area');
end

%Program for white balancing

function W=wbalance(im)
im2=im;
im1=rgb2ycbcr(im);
Lu=im1(:,:,1);
Cb=im1(:,:,2);
Cr=im1(:,:,3);
[x y z]=size(im);
tst=zeros(x,y);
Mb=sum(sum(Cb));
Mr=sum(sum(Cr));
Mb=Mb/(x*y);
Mr=Mr/(x*y);
Db=sum(sum(Cb-Mb))/(x*y);
Dr=sum(sum(Cr-Mr))/(x*y);
cnt=1;
Ciny = zeros(1,x*y);
for i=1:x
    for j=1:y
        b1=Cb(i,j)-(Mb+Db*sign(Mb));
        b2=Cr(i,j)-(1.5*Mr+Dr*sign(Mr));
        if (b1<(1.5*Db) & b2<(1.5*Dr))
            Ciny(cnt)=Lu(i,j);
            tst(i,j)=Lu(i,j);
            cnt=cnt+1;
        end
    end
end
cnt=cnt-1;
Ciny(cnt:end) = [];
iy=sort(Ciny,'descend');
nn=round(cnt/10);
Ciny2(1:nn)=iy(1:nn);
mn=min(Ciny2);
c=0;
for i=1:x
    for j=1:y
        if tst(i,j)<mn
            tst(i,j)=0;
        else
            tst(i,j)=1;
            c=c+1;
        end
    end
end
R=im(:,:,1);
G=im(:,:,2);
B=im(:,:,3);
R=double(R).*tst;
G=double(G).*tst;
B=double(B).*tst;
Rav=mean(mean(R));
Gav=mean(mean(G));
Bav=mean(mean(B));
Ymax=double(max(max(Lu)))/15;
Rgain=Ymax/Rav;
Ggain=Ymax/Gav;
Bgain=Ymax/Bav;
im(:,:,1)=im(:,:,1)*Rgain;
im(:,:,2)=im(:,:,2)*Ggain;
im(:,:,3)=im(:,:,3)*Bgain;
W=im;
end

