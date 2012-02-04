function Detection_Windows=detectorBB_Sliding_SVM(windowCandidates,im,model_HOG)
%% Detecta las window candidates que contienen una señal
% Aplica Sliding Windows, un clasificador y clustering para filtrar
% Input: windowCandidates - struct('x',[],'y',[],'w',[],'h',[]);
%        im - imagen
% Output: Detection_Windows - struct('x',[],'y',[],'w',[],'h',[]);
if (isempty(windowCandidates)==0)
for k = 1 : numel(windowCandidates)
    % Aplicamos sliding windows sobre cada bounding box
    sub_candidates = sub_sliding_windows(windowCandidates(k));
    disp ('SlidingWindows: '), disp(length(sub_candidates));
    % Por cada sub_sliding_window creado lo pasamos por el clasificador que nos devolvera las sub_sliding_windows válidas.
    % Para tomar el cuadrado más grande por ejemplo
    area = 0;
    j = 0;
    if (isempty(sub_candidates(1).x)==0)
        for c = 1 : length(sub_candidates)
            sub_window = im(int16(sub_candidates(c).y):int16(sub_candidates(c).y+sub_candidates(c).h), int16(sub_candidates(c).x):int16(sub_candidates(c).x+sub_candidates(c).w), :);
            sub_window = imresize(sub_window, [64 64]);
            % CALSIFICAMOS SUBWINDOW
            predicted_label=clasificate_subwindows_binary(sub_window,model_HOG); %model HOG viene del archivo
            if (predicted_label == 1)
                j = j + 1;
                % Guardo el area de la sub-ventana más grande, también podría hacer la media de las subventanas.
                if (sub_candidates(c).w * sub_candidates(c).h > area)
                    area = sub_candidates(c).w * sub_candidates(c).h;
                end
                % El código anterior tiene un fallo, si encontramos en una subventana dos o más señales, todas tendrán el mismo área de
                % cuadrado cuando probablemente dichas señales tengan tamaños diferentes. Esto pasa raras veces, pero pasa.
                if (j == 1)
                    sub_selected_candidates = [sub_candidates(c).x+sub_candidates(c).w/2;sub_candidates(c).y+sub_candidates(c).h/2];
                else
                    sub_selected_candidates = [sub_selected_candidates, [sub_candidates(c).x+sub_candidates(c).w/2;sub_candidates(c).y+sub_candidates(c).h/2]];
                end
%                 rectangle('Position',[sub_candidates(c).x sub_candidates(c).y sub_candidates(c).w sub_candidates(c).h],'EdgeColor','b');
            end
        end
    end
    % Si hay sub_sliding_windows válidas miramos a qué grupo pertenecen.
    Detection_Windows = struct('x',[],'y',[],'w',[],'h',[]);
    if (j > 0)
        % el clustering se puede mejorar, haciendolo sobre x,y,w,h en lugar
        % de sobre el centro, para evitar problemas con señales cercanas
        [clustCent,~,clustMembsCell] = MeanShiftCluster(sub_selected_candidates, 40);
        numClust = length(clustMembsCell);
        Detection_Windows(min(numClust)) = struct('x',[],'y',[],'w',[],'h',[]);
        for l = 1:min(numClust)
            myClustCen = clustCent(:,l);
            %                    rectangle('Position',[myClustCen(1)-20 myClustCen(2)-20 40 40],'EdgeColor','y');
            Detection_Windows(l).x = myClustCen(1)-sqrt(area)/2;
            Detection_Windows(l).y = myClustCen(2)-sqrt(area)/2;
            Detection_Windows(l).w = sqrt(area);
            Detection_Windows(l).h = sqrt(area);
        end
    end
end
else
    Detection_Windows=struct('x',[],'y',[],'w',[],'h',[]);
end

end

function predicted_label=clasificate_subwindows_binary(im,model_HOG)
%% Detecta si en la ventana (64x64) hay una señal o no la hay
imgray = double(rgb2gray(im));
H=HOG(imgray);
[predicted_label, accuracy, prob_estimates] = svmpredict( 0, H', model_HOG, '-b 1' );
end

function sub_windows = sub_sliding_windows(bounding)  %(bounding,image)
    sub_windows = struct('x',[],'y',[],'w',[],'h',[]);
        %First we make bounding box 10% larger in each direction
        maxY = 1236; 
        maxX =1628;
        x = bounding.x - round(bounding.w/20);
        y = bounding.y - round(bounding.h/20);
        w = bounding.w + bounding.w/10;
        h = bounding.h + bounding.h/10;
        if(x<1) x=1; end;
        if(y<1) y=1; end;
        if((x+w)>maxX)  w=maxX-x; end;
        if((y+h)>maxY)  h=maxY-y; end; 
        minsize = 40;                       %min size of the sliding window
        maxsize = min([250,min([w,h])]);    %max size of the sliding window
        jmpsize = (maxsize-minsize)/3;      %3 different sizes  
        contador = 0;
        for win_size=maxsize:-jmpsize:minsize
            jmpX = win_size/10;
            jmpY = win_size/10;
            for yy=y:jmpY:y+h-win_size
                for xx=x:jmpX:x+w-win_size
                   contador = contador + 1;
                   sub_windows(contador).x = xx;
                   sub_windows(contador).y = yy;
                   sub_windows(contador).w = win_size;
                   sub_windows(contador).h = win_size;
                end
            end
        end
end
