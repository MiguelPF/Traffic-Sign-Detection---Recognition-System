function [segmentedIm]=SegmentIm(im)
%% Generate the segmentedIm from the original im
% segmentedIm is a binary image
        %im = imread(strcat(directory,'/',files(i).name));
        % Extraemos un vector de pixels validos
        pixelCandidates = CandidateGenerationPixel_Color(im, 'hsv');
        maskFound = im(:,:,1) .* 0;
        maskFound(pixelCandidates) = 1;
        %morfologia
        %First we apply a opening with an small structuring element to
        %elimiate noise (isolated pixel or small regions)
        se = strel('disk',2);
        maskOpen = imopen(maskFound,se);
        %Then we apply a closing with a big structuring element to fill the
        %white and/or black region inside the signal and include it into
        %the mask
        se2 = strel('disk',20);
        maskClose = imclose(maskOpen,se2);
        BWdfill = imfill(maskClose, 'holes');   
        BWnobord = imclearborder(BWdfill, 4);
        seD = strel('diamond',1);
        BWfinal = imerode(BWnobord,seD);
        segmentedIm = imerode(BWfinal,seD);
end


function [pixelCandidates] = CandidateGenerationPixel_Color(im, space)
%% Generate Color im segmentation for traffic sign
im=double(im);
%im=INDANE(im);
switch space
    case 'normrgb'
        R=im(:,:,1);
        G=im(:,:,2);
        B=im(:,:,3);
        %Miguel
        pixelCandidates_red = (R > 0.2549) & (G(:,:)<0.57*R(:,:)) & (B(:,:)<0.57*R(:,:));
        pixelCandidates_blue = (B > 0.2549) & (G(:,:)<0.62*B(:,:)) & (R(:,:)<0.62*B(:,:));
        
        pixelCandidates = pixelCandidates_blue | pixelCandidates_red;
    case 'hsv'
        im=rgb2hsv(im);
        H=im(:,:,1);
        S=im(:,:,2);
        V=im(:,:,3);
        pixelCandidates_red = ((H>0.95)|(H<0.1)) & (V>40) & (S>0.55);
        pixelCandidates_blue = ((H>0.57)&(H<0.72)) & (V>35) & (S>0.35);
        pixelCandidates=pixelCandidates_blue | pixelCandidates_red;
    case 'opp'
        I=im(:,:,1)+im(:,:,2)+im(:,:,3);
        RminusG=im(:,:,1)-im(:,:,2);
        YminusB=im(:,:,1)+im(:,:,2)-2.*im(:,:,3);
        pixelCandidates_red = (I > 114 & I<260) & (RminusG>10 & RminusG<55) & (YminusB<60 & YminusB>0);
        pixelCandidates_blue = (I > 140 & I<260) & (RminusG<4) & (YminusB<2);
        pixelCandidates=pixelCandidates_blue | pixelCandidates_red;
    otherwise
        error('Incorrect color space defined');
        return
end
end