% 20160312
% depth based inpainting
% chinthakawk@gmail.com
%
% original concept is from the following link
% https://github.com/surahul/Inpaint/tree/master/Inpaint-qt
%
% depth term is from the following paper
% [5]	I. Daribo and H. Saito, ?A novel inpainting-based layered depth 
% video for 3DTV,? IEEE Transactions on Broadcasting, vol. 57, pp. 533-541,
% 2011.
%
% there can be still some errors in the code.

function inpaintDepthBasedFE()

tic;

clc; clear all;

fol = '';
name = '';
iColor = imread(strcat(fol,name,'_image.png'));
iDepth = imread(strcat(fol,name,'_depth.png'));
iMask = imread(strcat(fol,name,'_holes.png'));
halfPatchWidth = 4;
scale = .5;

fid = fopen(strcat(fol,name,'_log_',num2str(scale),'.txt'),'w');
fprintf(fid,datestr(now,'mm/dd/yyyy HH:MM:SS AM'));
fprintf(fid,'\n\nfolder: %s',fol);
fprintf(fid,'\nname: %s',name);
fprintf(fid,'\nhalf patch width: %d',halfPatchWidth);

fprintf(fid,'\n\noriginal image size = (%d,%d)',size(iMask));

iColor = imresize(iColor,scale);
iDepth = imresize(iDepth,scale);
iMask = imresize(iMask,scale);

fprintf(fid,'\nscale used = %f',scale);
fprintf(fid,'\nscaled image size = (%d,%d)',size(iMask));
fprintf('\n\nsize(iColor) = (%d,%d)',size(iMask));

% figure;
% subplot(1,3,1); imshow(iColor);
% subplot(1,3,2); imshow(iDepth);
% subplot(1,3,3); imshow(iMask);

% imshow(iColor);

% =============
% initializeMats
workImage = iColor;
depth = iDepth;

confidence = double(iMask);
% figure; imshow(confidence); title('confidence');

sourceRegion = iMask;
updatedMask = iMask;
originalSourceRegion = sourceRegion;

targetRegion = imcomplement(iMask);
% figure; imshow(targetRegion); title('targetRegion');

data = double(zeros(size(iMask)));
level = double(zeros(size(iMask)));

global LAPLACIAN_KERNEL;
LAPLACIAN_KERNEL = double(ones(3,3));
LAPLACIAN_KERNEL(2,2) = -8;
NORMAL_KERNELX = double(zeros(3,3));
NORMAL_KERNELX(1,2) = -1;
NORMAL_KERNELX(3,2) = 1;
NORMAL_KERNELY = NORMAL_KERNELX';

% =============
% calculateGradients
srcGray = rgb2gray(iColor);

[gradientX, gradientY] = imgradientxy(srcGray);

gradientX(sourceRegion==0) = 0;
gradientY(sourceRegion==0) = 0;

gradientX = gradientX/255;
gradientY = gradientY/255;
% fprintf('\nsize(gradientX) = (%d,%d)',size(gradientX));
% fprintf('\nsize(gradientY) = (%d,%d)',size(gradientY));

% figure; imshow(gradientX);
% figure; imshow(gradientY);
figure;
hax1 = axes;
imshow(iMask);
hold on;

figure;
hax2 = axes;
imshow(iColor);
hold on;

axes(hax1);

stay = true;
ite = 0;

fprintf(fid,'\n\n1.\titeration');
fprintf(fid,'\n2.\tholes pixel count');
fprintf(fid,'\n3.\tfill front count');
fprintf(fid,'\n4.\ttarge index');
fprintf(fid,'\n5.\ttarget index (row, column)');
fprintf(fid,'\n6.\tconfidence');
fprintf(fid,'\n7.\tdata');
fprintf(fid,'\n8.\tlevel');
fprintf(fid,'\n9.\tmax priority');
fprintf(fid,'\n10.\tsource patch upper left (row, column)');
fprintf(fid,'\n11.\tsource patch lower right (row, column)');
fprintf(fid,'\n12.\tpatch (width, height)');
fprintf(fid,'\n13.\tnumber of pixels copied\n');

% while stay
% if stay == true
while true
    
    imwrite(workImage,strcat(fol,name,'_workImage_',num2str(scale),...
        '_',num2str(ite),'.png'));
    imwrite(sourceRegion,strcat(fol,name,'_sourceRegion_',...
        num2str(scale),'_',num2str(ite),'.png'));
    imwrite(targetRegion,strcat(fol,name,'_targetRegion_',...
        num2str(scale),'_',num2str(ite),'.png'));
    imwrite(updatedMask,strcat(fol,name,'_updatedMask_',num2str(scale),...
        '_',num2str(ite),'.png'));
    
    ite = ite+1;
    pc = 0; % filled pixel count
    
    fprintf('\n%d:',ite);
    fprintf(fid,'\n%d:',ite);
    fprintf('\t%d',sum(sourceRegion(:)==0));
    fprintf(fid,'\t%d',sum(sourceRegion(:)==0));
    
    [fillFront, normals] = ...
        computeFillFront(LAPLACIAN_KERNEL,targetRegion,...
        NORMAL_KERNELX,NORMAL_KERNELY,sourceRegion);
%     disp(fillFront);
    
    fprintf('\t%d',size(fillFront,1));
    fprintf(fid,'\t%d',size(fillFront,1));
    
    confidence = computeConfidence(fillFront,halfPatchWidth,...
        workImage,targetRegion,confidence);
%     figure; imagesc(confidence); title('confidence');
%     fprintf('\nconfidence calculated');
%     fprintf('\nsize(confidence) = (%d,%d)',size(confidence));
    
    data = computeData(fillFront,normals,gradientX,gradientY);
%     figure; imagesc(data);
%     fprintf('\nsize(data) = (%d,%d)',size(data));
%     imshowpair(data,confidence);

    level = computeLevel(fillFront,halfPatchWidth,...
        workImage,targetRegion,level,depth);
%     figure; imagesc(level);
%     fprintf('\nsize(level) = (%d,%d)',size(level));
    
    [targetIndex,maxPriority] = ...
        computeTarget(fillFront,data,confidence,level);
    
    fprintf('\t%d',targetIndex);
    fprintf(fid,'\t%d',targetIndex);
    rt = fillFront(targetIndex,1);
    ct = fillFront(targetIndex,2);
    fprintf('\t(%d,%d)',rt,ct);
    fprintf(fid,'\t(%d,%d)',rt,ct);
    fprintf('\t%f',confidence(rt,ct));
    fprintf(fid,'\t%f',confidence(rt,ct));
    fprintf('\t%f',data(rt,ct));
    fprintf(fid,'\t%f',data(rt,ct));
    fprintf('\t%f',level(rt,ct));
    fprintf(fid,'\t%f',level(rt,ct));
    fprintf('\t%f',maxPriority);
    fprintf(fid,'\t%f',maxPriority);
    
    [bestMatchUpperLeft,bestMatchLowerRight] = ...
        computeBestPatch(fillFront,targetIndex,workImage,...
        originalSourceRegion,sourceRegion,halfPatchWidth);
    rsul = bestMatchUpperLeft(1,1);
    csul = bestMatchUpperLeft(1,2);
    rslr = bestMatchLowerRight(1,1);
    cslr = bestMatchLowerRight(1,2);
    
    fprintf('\t(%d,%d)',rsul,csul);
    fprintf(fid,'\t(%d,%d)',rsul,csul);
    fprintf('\t(%d,%d)',rslr,cslr);
    fprintf(fid,'\t(%d,%d)',rslr,cslr);
    
    [workImage,gradientX,gradientY,confidence,sourceRegion,targetRegion,...
        updatedMask,width,height,targetUpperLeft,pc] = ...
        updateMats(targetIndex,fillFront,halfPatchWidth,workImage,...
        sourceRegion,bestMatchUpperLeft,gradientX,gradientY,...
        confidence,targetRegion,updatedMask,pc,ite,fol,name,scale);
%     fprintf('\nupdateMats done');
    
%     figure; imshow(updatedMask); drawnow;
%     figure; imshow(workImage); drawnow;
%     hold on;
%     axes(hax1);
%     imshow(workImage);
    fprintf('\t(%d,%d)',width,height);
    fprintf(fid,'\t(%d,%d)',width,height);
    fprintf('\t%d',pc);
    fprintf(fid,'\t%d',pc);
    
    rtul = targetUpperLeft(1,1);
    ctul = targetUpperLeft(1,2);
    
    axes(hax1);
    rectangle('Position', [csul rsul width height],'EdgeColor','r');
    drawnow;
    axes(hax1);
    rectangle('Position', [ctul rtul width height],'EdgeColor','b');
    drawnow;
%     line([csul,ctul],[rsul,rtul]);
    dp = [ctul rtul] - [csul rsul];
    axes(hax1);
    q1 = quiver(csul,rsul,dp(1),dp(2),0,'MaxHeadSize',0.1);
    q1.Color = 'blue';
%     annotation('textarrow',[csul ctul]/size(workImage,2),[rsul rtul]/size(workImage,1),'Color','blue');
    drawnow;
    axes(hax1);
    text((csul+ctul)/2,(rsul+rtul)/2,num2str(ite));
    drawnow;
    
    f=getframe(hax1);
    [X, ~] = frame2im(f);
    imwrite(X,strcat(fol,name,'_markedMask_',num2str(scale),'_',...
        num2str(ite),'.png'),'png');
    
    axes(hax2);
    rectangle('Position', [csul rsul width height],'EdgeColor','r');
    drawnow;
    axes(hax2);
    rectangle('Position', [ctul rtul width height],'EdgeColor','b');
    drawnow;
%     line([csul,ctul],[rsul,rtul]);
    dp = [ctul rtul] - [csul rsul];
    axes(hax2);
    q2 = quiver(csul,rsul,dp(1),dp(2),0,'MaxHeadSize',0.1);
    q2.Color = 'blue';
%     annotation('textarrow',[csul ctul]/size(workImage,2),[rsul rtul]/size(workImage,1),'Color','blue');
    drawnow;
    axes(hax2);
    text((csul+ctul)/2,(rsul+rtul)/2,num2str(ite));
    drawnow;
    
    f=getframe(hax2);
    [X, ~] = frame2im(f);
    imwrite(X,strcat(fol,name,'_markedColor_',num2str(scale),...
        '_',num2str(ite),'.png'),'png');
    
%     imwrite(targetPatchColorBefore,strcat(fol,name,'_targetPatchColorBefore_',num2str(ite),'.png'));
    
    
    stay = checkEnd(sourceRegion);
    if stay==false
        break;
    end
    
    
    
    
    
%     imshow(workImage);
%     imagesc(updatedMask);
%     cv::imshow("updatedMask",updatedMask);
%         cv::imshow("inpaint",workImage);
%         cv::imshow("gradientX",gradientX);
%         cv::imshow("gradientY",gradientY);
%         cv::waitKey(2);
%     result=workImage.clone();
%     cv::namedWindow("confidence");
%     cv::imshow("confidence",confidence);

% cv::imwrite("result.jpg",i.result);
%             cv::namedWindow("result");
%             cv::imshow("result",i.result);
    
%     break;
end

f=getframe(hax1);
[X, ~] = frame2im(f);
imwrite(X,strcat(fol,name,'_markedMask_',num2str(scale),'.png'),'png');

f=getframe(hax2);
[X, ~] = frame2im(f);
imwrite(X,strcat(fol,name,'_markedColor_',num2str(scale),'.png'),'png');

toc;
% fprintf('\n\n%s',datestr(toc,'mm/dd/yyyy HH:MM:SS AM'));
fprintf(fid,'\n\nElapsed time is %f seconds',toc);

fclose(fid);
fprintf('\n');

end

function [fillFront, normals] = computeFillFront(LAPLACIAN_KERNEL,...
    targetRegion,NORMAL_KERNELX,NORMAL_KERNELY,sourceRegion)

% disp('computeFillFront');

boundryMat = filter2(LAPLACIAN_KERNEL,targetRegion,'same');
% fprintf('\nsize(boundryMat) = (%d,%d)',size(boundryMat));
sourceGradientX = filter2(NORMAL_KERNELX,sourceRegion,'same');
sourceGradientY = filter2(NORMAL_KERNELY,sourceRegion,'same');

% fillFront.clear();
% normals.clear();

% for(int x=0;x<boundryMat.cols;x++){
%     for(int y=0;y<boundryMat.rows;y++){
fi = 0;
for r = 1:size(boundryMat,1)
    for c = 1:size(boundryMat,2)
        if boundryMat(r,c)>0
            fi = fi+1;
            fillFront(fi,:) = [r,c];
            
            dx = sourceGradientX(r,c);
            dy = sourceGradientY(r,c);
            normal = [dy,-dx];
            tempF = sqrt((normal(1,1)*normal(1,1))+...
                (normal(1,2)*normal(1,2)));
            
            if tempF ~= 0
                normal(1,1) = normal(1,1)/tempF;
                normal(1,2) = normal(1,2)/tempF;
            end
            normals(fi,:) = normal;
        end
    end
end
% fprintf('\nfillFront size = %d',fi);

end

function confidence = computeConfidence(fillFront,halfPatchWidth,...
    workImage,targetRegion,confidence)
%     fprintf('\nsize(fillFront) = (%d,%d)',size(fillFront));
%     figure; imshow(confidence); title('confidence');
    for i = 1:size(fillFront,1)
        currentPoint = fillFront(i,:);
        [a, b] = getPatch(currentPoint,halfPatchWidth,workImage);
        total = 0.0;
%         for (int x=a.x;x<=b.x;x++){
%             for(int y=a.y;y<=b.y;y++){
%         fprintf('\n%d\t%d\t%d',i,a(1,1),b(1,1));
        for r = a(1,1):b(1,1)
            for c = a(1,2):b(1,2)
            	if targetRegion(r,c) == 0
%                     fprintf('\ntarget region is 0');
                    total = total + confidence(r,c);
%                     fprintf('\n%d\t%f\t%f',i,confidence(r,c),total);
                end
            end
        end
        
        confidence(currentPoint(1,1),currentPoint(1,2)) = ...
            total/((b(1,2)-a(1,2)+1)*(b(1,1)-a(1,1)+1));
    end
%     fprintf('\nlast i of fillFront = %d',i);
end

function level = computeLevel(fillFront,halfPatchWidth,...
    workImage,targetRegion,level,depth)
%     fprintf('\nsize(fillFront) = (%d,%d)',size(fillFront));
%     figure; imshow(confidence); title('confidence');
%     fprintf('\ninside computeLevel');
    for i = 1:size(fillFront,1)
        currentPoint = fillFront(i,:);
        [a, b] = getPatch(currentPoint,halfPatchWidth,workImage);
%         total = 0.0;
        z = 0;
%         fprintf('\n%d',i);
%         for (int x=a.x;x<=b.x;x++){
%             for(int y=a.y;y<=b.y;y++){
%         fprintf('\n%d\t%d\t%d',i,a(1,1),b(1,1));
        for r = a(1,1):b(1,1)
            for c = a(1,2):b(1,2)
%                 fprintf('\n%d\t%d\t%d',i,r,c);
%                 fprintf('\n%d\t%f',i,targetRegion(r,c));
            	if targetRegion(r,c) == 0
%                     fprintf('\ntarget region is 0');
                    z = z+1;
                    d(z,1) = depth(r,c);
%                     total = total + confidence(r,c);
%                     fprintf('\n%d\t%f\t%f',i,confidence(r,c),total);
                end
            end
        end
        
        dm = mean(d);
        
        level(currentPoint(1,1),currentPoint(1,2)) = ...
            z / (z + sum((d-dm).^2) );
    end
%     fprintf('\nlast i of fillFront = %d',i);
end

function [upperLeft, lowerRight] = ...
    getPatch(centerPixel,halfPatchWidth,workImage)
    c = centerPixel(1,2);
    r = centerPixel(1,1);

    minC = max(c-halfPatchWidth,1);
    maxC = min(c+halfPatchWidth,size(workImage,2));
    minR = max(r-halfPatchWidth,1);
    maxR = min(r+halfPatchWidth,size(workImage,1));

    upperLeft(1,2) = minC;
    upperLeft(1,1) = minR;

    lowerRight(1,2) = maxC;
    lowerRight(1,1) = maxR;
end

function data = computeData(fillFront,normals,gradientX,gradientY)
    for i = 1:size(fillFront,1)
        currentPoint = fillFront(i,:);
        currentNormal = normals(i,:);
        data(currentPoint(1,1),currentPoint(1,2)) =...
            abs(gradientX(currentPoint(1,1),currentPoint(1,2))*...
            currentNormal(1,1)+...
            gradientY(currentPoint(1,1),currentPoint(1,2))*...
            currentNormal(1,1))+.001;
    end
end

function [targetIndex,maxPriority] = ...
    computeTarget(fillFront,data,confidence,level)
    targetIndex = 1;
    maxPriority = 0;
    priority = 0;
%     cv::Point2i currentPoint;
    for i = 1:size(fillFront,1)
        currentPoint = fillFront(i,:);
        priority = data(currentPoint(1,1),currentPoint(1,2))*...
            confidence(currentPoint(1,1),currentPoint(1,2))*...
            level(currentPoint(1,1),currentPoint(1,2));
%         fprintf('\n%d:\t%f\t%f\t%f',i,...
%             data(currentPoint(1,1),currentPoint(1,2)),...
%             confidence(currentPoint(1,1),currentPoint(1,2)),priority);
        if priority > maxPriority
            maxPriority = priority;
            targetIndex = i;
        end
    end
end

function [bestMatchUpperLeft,bestMatchLowerRight] = ...
    computeBestPatch(fillFront,targetIndex,workImage,...
    originalSourceRegion,sourceRegion,halfPatchWidth)
%     fprintf('\ninside computeBestPatch');
    minError = 9999999999999999;
    bestPatchVarience = 9999999999999999;
    currentPoint = fillFront(targetIndex,:);
    
%     cv::Vec3b sourcePixel,targetPixel;
%     double meanR,meanG,meanB;
%     double difference,patchError;
%     bool skipPatch;
    [a, b] = getPatch(currentPoint,halfPatchWidth,workImage);

    width = b(1,2)-a(1,2)+1;
    height = b(1,1)-a(1,1)+1;
%     for(int x=0;x<=workImage.cols-width;x++){
%         for(int y=0;y<=workImage.rows-height;y++){
    fprintf('\n');
    for r = 1:size(workImage,1)-height+1
        fprintf('.');
        for c = 1:size(workImage,2)-width+1
            patchError = 0;
            meanR = 0; meanG = 0; meanB = 0;
            skipPatch = false;

%             for(int x2=0;x2<width;x2++){
%                 for(int y2=0;y2<height;y2++){
            for r2 = 0:height-1
                for c2 = 0:width-1
                    if originalSourceRegion(r+r2,c+c2) == 0
                        skipPatch = true;
                        break;
                    end
                    
                    % start
%                     if ((a(1,1)+r2)>=size(sourceRegion,1) ||...
%                             (a(1,2)+c2)>=size(sourceRegion,2))
%                         continue;
%                     end
                    % end
                    
                    if sourceRegion(a(1,1)+r2,a(1,2)+c2)==0
                        continue;
                    end

                    sourcePixel = workImage(r+r2,c+c2,:);
                    targetPixel = workImage(a(1,1)+r2,a(1,2)+c2,:);

                    for i = 1:3
                        difference = sourcePixel(1,i)-targetPixel(1,i);
                        patchError = patchError + difference*difference;
                    end
                    meanB = meanB + sourcePixel(1,1);
                    meanG = meanG + sourcePixel(1,2);
                    meanR = meanR + sourcePixel(1,3);
                end
                    
                if(skipPatch)
                    break;
                end
            end

            if(skipPatch)
                continue;
            end
            if patchError<minError
                minError = patchError;
                bestMatchUpperLeft = [r,c];
                bestMatchLowerRight = [r+height-1, c+width-1];

                patchVarience = 0;    
                
%                 % my code start
%                 if ((a(1,1)+r2)>=size(sourceRegion,1) || (a(1,2)+c2)>=size(sourceRegion,2))
%                     continue;
%                 end
%                 % end
                
                for r2 = 0:height-1
                    for c2 = 0:width-1
%                         if ((a(1,1)+r2)>=size(sourceRegion,1) ||...
%                                 (a(1,2)+c2)>=size(sourceRegion,2))
%                             continue;
%                         end
                        if sourceRegion(a(1,1)+r2,a(1,2)+c2) == 0
                            sourcePixel=workImage(r+r2,c+c2,:);
%                             disp(sourcePixel);
                            difference=sourcePixel(1,1)-meanB;
                            patchVarience=...
                                patchVarience+difference*difference;
                            difference=...
                                sourcePixel(1,2)-meanG;
                            patchVarience=...
                                patchVarience+difference*difference;
                            difference=sourcePixel(1,3)-meanR;
                            patchVarience=...
                                patchVarience+difference*difference;
                        end
                    end
                end
                bestPatchVarience=patchVarience;

            elseif(patchError==minError)
                patchVarience=0;
                
                % start
%                 if ((a(1,1)+r2)>=size(sourceRegion,1) || (a(1,2)+c2)>=size(sourceRegion,2))
%                     continue;
%                 end
                % end
                
                
                for r2=0:height-1
                    for c2=0:width-1
%                         if ((a(1,1)+r2)>=size(sourceRegion,1) ||...
%                                 (a(1,2)+c2)>=size(sourceRegion,2))
%                             continue;
                        if sourceRegion(a(1,1)+r2,a(1,2)+c2)==0
                            sourcePixel=workImage(r+r2,c+c2,:);
                            difference=sourcePixel(1,1)-meanB;
                            patchVarience=...
                                patchVarience+difference*difference;
                            difference=...
                                sourcePixel(1,2)-meanG;
                            patchVarience=...
                                patchVarience+difference*difference;
                            difference=...
                                sourcePixel(1,3)-meanR;
                            patchVarience=...
                                patchVarience+difference*difference;
                        end
                    end
                end
                if(patchVarience<bestPatchVarience)
                    minError=patchError;
                    bestMatchUpperLeft=[r,c];
                    bestMatchLowerRight=[r+height-1,c+width-1];
                    bestPatchVarience=patchVarience;
                end
            end
        end
    end
end

function [workImage,gradientX,gradientY,confidence,sourceRegion,...
    targetRegion,updatedMask,width,height,targetUpperLeft,pc]=...
    updateMats(targetIndex,fillFront,halfPatchWidth,workImage,....
    sourceRegion,bestMatchUpperLeft,gradientX,gradientY,confidence,...
    targetRegion,updatedMask,pc,ite,fol,name,scale)
    
    targetPoint=fillFront(targetIndex,:);
    [a,b]=getPatch(targetPoint,halfPatchWidth,workImage);
    targetUpperLeft = a;
    width=b(1,2)-a(1,2)+1;
    height=b(1,1)-a(1,1)+1;
    
    sourcePatchColor = workImage(max(bestMatchUpperLeft(1,1),1):...
        min(bestMatchUpperLeft(1,1)+height,size(workImage,1)),...
        max(bestMatchUpperLeft(1,2),1):...
        min(bestMatchUpperLeft(1,2)+width,size(workImage,2)),:);
    imwrite(sourcePatchColor,strcat(fol,name,'_sourcePatchColor_',...
        num2str(scale),'_',num2str(ite),'.png'));
    
    targetPatchColorBefore = workImage(max(targetUpperLeft(1,1),1):...
        min(targetUpperLeft(1,1)+height,size(workImage,1)),...
        max(targetUpperLeft(1,2),1):...
        min(targetUpperLeft(1,2)+width,size(workImage,2)),:);
    imwrite(targetPatchColorBefore,strcat(fol,name,...
        '_targetPatchColorBefore_',num2str(scale),'_',num2str(ite),'.png'));
    
%     disp(size(updatedMask));
%     fprintf('\n[%d:%d,%d:%d]',targetUpperLeft(1,1),targetUpperLeft(1,1)+height,...
%                     targetUpperLeft(1,2),targetUpperLeft(1,2)+width);
    
    holesPatch = updatedMask(max(targetUpperLeft(1,1),1):...
        min(targetUpperLeft(1,1)+height,size(updatedMask,1)),...
        max(targetUpperLeft(1,2),1):...
        min(targetUpperLeft(1,2)+width,size(updatedMask,2)));
    imwrite(holesPatch,strcat(fol,name,'_holesPatch_',...
        num2str(scale),'_',num2str(ite),'.png'));

    holesPatchSource = zeros(size(holesPatch,1),size(holesPatch,2),3,...
        'uint8');
    holesPatchSource(:,:,:) = 255;
    
%     for(int x=0;x<width;x++){
%         for(int y=0;y<height;y++){
    for r=0:height-1
        for c=0:width-1
            % start
%             if ((a(1,1)+r)>=size(sourceRegion,1) ||...
%                     (a(1,2)+c)>=size(sourceRegion,2))
%                 continue;
%             end
            % end
            
            if(sourceRegion(a(1,1)+r,a(1,2)+c)==0)
                pc = pc+1;
%                 workImage.at<cv::Vec3b>(a.y+y,a.x+x)=workImage.at<cv::Vec3b>(bestMatchUpperLeft.y+y,bestMatchUpperLeft.x+x);
                workImage(a(1,1)+r,a(1,2)+c,:)=...
                    workImage(bestMatchUpperLeft(1,1)+r,...
                    bestMatchUpperLeft(1,2)+c,:);
                gradientX(a(1,1)+r,a(1,2)+c)=...
                    gradientX(bestMatchUpperLeft(1,1)+r,...
                    bestMatchUpperLeft(1,2)+c);
                gradientY(a(1,1)+r,a(1,2)+c)=...
                    gradientY(bestMatchUpperLeft(1,1)+r,...
                    bestMatchUpperLeft(1,2)+c);
                confidence(a(1,1)+r,a(1,2)+c)=...
                    confidence(targetPoint(1,1),targetPoint(1,2));
                sourceRegion(a(1,1)+r,a(1,2)+c)=1;
                targetRegion(a(1,1)+r,a(1,2)+c)=0;
                updatedMask(a(1,1)+r,a(1,2)+c)=0;
                holesPatchSource(r+1,c+1,:) =...
                    workImage(bestMatchUpperLeft(1,1)+r,...
                    bestMatchUpperLeft(1,2)+c,:);
            end
        end
    end
    
    targetPatchColorAfter = workImage(max(targetUpperLeft(1,1),1):...
        min(targetUpperLeft(1,1)+height,size(workImage,1)),...
        max(targetUpperLeft(1,2),1):...
        min(targetUpperLeft(1,2)+width,size(workImage,2)),:);
    imwrite(targetPatchColorAfter,strcat(fol,name,...
        '_targetPatchColorAfter_',num2str(scale),'_',num2str(ite),'.png'));
    
    imwrite(holesPatchSource,strcat(fol,name,'_holesPatchSource_',...
        num2str(scale),'_',num2str(ite),'.png'));
end

function stay = checkEnd(sourceRegion)
%     for r=1:size(sourceRegion,1)
%         for c=1:size(sourceRegion,2)
% %     for(int x=0;x<sourceRegion.cols;x++){
% %         for(int y=0;y<sourceRegion.rows;y++){
%             if(sourceRegion(r,c)==0)
%                 stay = true;
%             else
%                 stay = false;
%             end
%         end
%     end
    if sum(sourceRegion(:)==0)
        stay = true;
    else
        stay = false;
    end
%     fprintf('\nholes = %d\n',sum(sourceRegion(:)==0));
%     disp(stay);
end