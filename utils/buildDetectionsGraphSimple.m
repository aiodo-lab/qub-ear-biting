% Copyright (c) 2015 Niall McLaughlin, CSIT, Queen's University Belfast, UK
% Contact: nmclaughlin02@qub.ac.uk
% If you use this code please cite:
% "Enhancing Linear Programming with Motion Modeling for Multi-target Tracking",
% N McLaughlin, J Martinez Del Rincon, P Miller, 
% IEEE Winter Conference on Applications of Computer Vision (WACV), 2015 
% 
% This software is licensed for research and non-commercial use only.
% 
% The above copyright notice and this permission notice shall be included in
% all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
% THE SOFTWARE.


%return a linking graph - indicates which detections are linked together
%linkGraph(i,j) = cost of linking detection i with detection j
%linkIndexGraph = index refering to the edge between detections i and j -
%                 zero if no link is possible
%detections = sorted and filtered detections between startFrame and
%             endFrame
%appearanceModels = sorted and filtered appearances between startFrame and
%             endFrame
function [linkGraph,linkIndexGraph,nTotalLinks,detections,appearanceModels] = buildDetectionsGraphSimple(detections,startFrame,endFrame,maxFrameGap,heightsPerSec,appearanceThreshold,fps,appearanceModels)

    [~,inds] = sort(detections(:,1),'ascend');
    detections = detections(inds,:);
    appearanceModels = appearanceModels(inds,:);

    allInds = find(detections(:,1) >= startFrame & detections(:,1) <= endFrame);
    detections = detections(allInds,:);
    appearanceModels = appearanceModels(allInds,:);
    nTotalDetections = size(detections,1);
    detections = [detections'; 1:size(detections,1)]';
    
    linkGraph = ones(nTotalDetections,nTotalDetections) * -1;
    linkIndexGraph = zeros(nTotalDetections,nTotalDetections);

    prevFrameDetections = [];
    nPrevFrameDetections = 0;
    
    linkIndex = 1;   

    for f = startFrame:endFrame

        thisFrameDetections = detections(find(detections(:,1) == f),:);
        thisFrameDetections = round(thisFrameDetections);
        nThisFrameDetections = size(thisFrameDetections,1);

        if nPrevFrameDetections > 0

            %find the distances
            for i = 1:nPrevFrameDetections
                for j = 1:nThisFrameDetections
                    
                    distBetweenDets = sqrt(sum((thisFrameDetections(j,3:4) - prevFrameDetections(i,3:4)).^2));

                    % row vectors for x and y in this frame
                    thisX = [thisFrameDetections(j,3:6)];
                    thisY = [thisFrameDetections(j,7:10)];
                   
                    % row vectors for x and y in previous frame
                    prevX = [prevFrameDetections(i,3:6)];
                    prevY = [prevFrameDetections(i,7:10)];

                    % area of polygons (prev. and current frame)
                    areathisPoly = polyarea(thisX, thisY);
                    areaprevPoly = polyarea(prevX, prevY);

                    poly1 = polyshape([thisFrameDetections(j,3:6)], [thisFrameDetections(j,7:10)]);
                    poly2 = polyshape([prevFrameDetections(i,3:6)], [prevFrameDetections(i,7:10)]);

                    % area of the intersection between polygon in this
                    % frame and previous frame
                    intersection = intersect(poly1,poly2);
                    areaIntersection = area(intersection);
                    
                    % add areas of prev and current polygons less
                    % intersection then compute iou.
                    union = (areathisPoly + areaprevPoly) - areaIntersection;
                    detectionOverlap = areaIntersection / union;
    
                    %intersection = rectint([ 0 0 thisFrameDetections(j,5:6)],[0 0 prevFrameDetections(i,5:6)]);
                    %union = prod(thisFrameDetections(j,5:6)) + prod(prevFrameDetections(i,5:6)) - intersection;
                    %detectionOverlap = intersection / union;
                    
                    timeGap = thisFrameDetections(j,1) - prevFrameDetections(i,1);

                    DetNumPrev = prevFrameDetections(i,12); % from 8 - 12
                    DetNumThis = thisFrameDetections(j,12); % from 8 - 12
                    costAppearance = 1 - sum(sqrt(appearanceModels(DetNumPrev,:) .* appearanceModels(DetNumThis,:)));                     
                                                             
                    %max allowed speed is relative to the bounding box size
                    maxSpeed = (thisFrameDetections(j,10) * heightsPerSec) / fps; % changed index from 6 - 10
                    DistanceThreshold = maxSpeed * timeGap;
                    
                    if timeGap > 0 && timeGap <= maxFrameGap && detectionOverlap >= 0.5 && distBetweenDets <= DistanceThreshold && costAppearance <= appearanceThreshold
                        
                        costDist = 1 - exp(1).^-((distBetweenDets / DistanceThreshold)^2);
                        costTime = 1 - exp(1).^-(((timeGap-1) / maxFrameGap)^2);                                                                                                                
 
                        linkGraph(prevFrameDetections(i,12),thisFrameDetections(j,12)) = costAppearance + costDist + costTime; % changed index from 8 - 12
                        linkIndexGraph(prevFrameDetections(i,12),thisFrameDetections(j,12)) = linkIndex; % changed index from 8 - 12
                        linkIndex = linkIndex + 1;
                    end
                end
            end
        end

        prevFrameDetections = [prevFrameDetections; thisFrameDetections];
        inds = find((f - prevFrameDetections(:,1)) < maxFrameGap);
        prevFrameDetections = prevFrameDetections(inds,:);
        nPrevFrameDetections = size(prevFrameDetections,1);
    end

    nTotalLinks = sum(linkGraph(:) ~= -1);    
end
