%% Topology Optimization Program 
% Based on template code "top88".
clear; close all
% Debugging inputs
nelx=30;
nely=10;
volfrac=.5;
penal=3;
rmin=1.5;
ft=1;
framecount=0;

for i=1:nelx
    loc=1+(nely*(i-1));  % Sweep across top beam... so 1, nely+1, 2*nely+1, 3*nely+1...
    top_solve(nelx,nely,volfrac,penal,rmin,ft,loc);
    framecount=framecount+1;
    %frames(framecount)=getframe;
    im{framecount}=frame2im(getframe);
end


%% Outputting as GIF

%v=VideoWriter('mov');open(v);writeVideo(v,frames);close(v);

filename = 'testAnimated.gif';
for idx = 1:framecount
    [A,map] = rgb2ind(im{idx},256);
    if idx == 1
        imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',.5);
    else
        imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',.5);
    end
end

%% Optimization Problem Solver
function top_solve(nelx,nely,volfrac,penal,rmin,ft,loc) % output frames
%% MATERIAL PROPERTIES
E0 = 1;
Emin = 1e-9; % Some small number.
nu = 0.3;


%% PREPARE for FINITE ELEMENT ANALYSIS
% stiffness matrix
A11 = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12];
A12 = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6];
B11 = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4];
B12 = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2];

KE = 1/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11]);

nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1);
edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1);
iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1);
jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1);

%% DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)
F = sparse(2*(nely+1)*(nelx+1),1); F(loc*2,1) = -1; % Locate the y-component of the "LOC" node
U = sparse(2*(nely+1)*(nelx+1),1);
fixeddofs = union([1:2:2*(nely+1)],[2*(nelx+1)*(nely+1)]); % Which nodes are fixed in X DIRECTION, and which are fixed in Y DIRECTION.
alldofs = [1:2*(nely+1)*(nelx+1)]; % All possible nodes.
freedofs = setdiff(alldofs,fixeddofs); % Which ndoes are NOT fixed

%% PREPARE FILTER
% Smooth the solution to make it more manufacturable.
iH = ones(nelx*nely*(2*(ceil(rmin)-1)+1)^2,1);
jH = ones(size(iH));
sH = zeros(size(iH));
k = 0;
for i1 = 1:nelx
  for j1 = 1:nely
    e1 = (i1-1)*nely+j1;
    for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
      for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
        e2 = (i2-1)*nely+j2;
        k = k+1;
        iH(k) = e1;
        jH(k) = e2;
        sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));
      end
    end
  end
end
H = sparse(iH,jH,sH);
Hs = sum(H,2);

%% INITIALIZE ITERATION
x = repmat(volfrac,nely,nelx); % Starting x!
xPhys = x;
loop = 0;
change = 1;

%% START ITERATION
while change > 0.01
  loop = loop + 1; % Update the count.
  
  % STEP 3.1) FE-ANALYSIS
    % KE, F is defined above
    K = sparse(2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1));
    
    
    for ely = 1:nely
        for elx = 1:nelx
            n1 = (nely+1)*(elx-1)+ely;
            n2 = (nely+1)* elx +ely;
            edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 
                2*n2+1;2*n2+2;2*n1+1; 2*n1+2]; 
            K(edof,edof) = K(edof,edof) + x(ely,elx)^penal*KE;
        end
    end
    
    U(freedofs,:) = K(freedofs,freedofs)\F(freedofs,:);
    U(fixeddofs,:)= 0;
  
  %% OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    c=0;
    for ely = 1:nely
            for elx = 1:nelx
                n1 = (nely+1)*(elx-1)+ely;
                n2 = (nely+1)* elx +ely;
                Ue = U([2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2],1);
                ce = x(ely,elx)^penal*Ue'*KE*Ue; % Element strain energy
                c = c + ce; % Total strain energy
                dc(ely,elx) = -penal*x(ely,elx)^(penal-1)*Ue'*KE*Ue; % Design sensitivity
            end
    end
    dv = ones(nely,nelx);

  %% FILTERING/MODIFICATION OF SENSITIVITIES. To make it more realistic.
    if ft == 1
        dc(:) = H*(x(:).*dc(:))./Hs./max(1e-3,x(:));
    elseif ft == 2
        dc(:) = H*(dc(:)./Hs);
        dv(:) = H*(dv(:)./Hs);
    end

  %% OPTIMALITY CRITERIA UPDATE OF DESIGN VARIABLES AND PHYSICAL DENSITIES
    l1 = 0; l2 = 100000; move = 0.2;
    while (l2-l1 > 1e-4)
        lmid = 0.5*(l2+l1); % Get the average
        xnew = max(0.001,max(x-move,min(1.,min(x+move,x.*sqrt(-dc./lmid)))));
        if sum(sum(xnew)) - volfrac*nelx*nely > 0
            l1 = lmid;
        else
            l2 = lmid;
        end
    end
  change = max(abs(xnew(:)-x(:)));
  x = xnew;

  %% PRINT RESULTS
  %fprintf(' It.:%5i Obj.:%11.4f Vol.:%7.3f ch.:%7.3f\n',loop,c, ...
   % mean(x(:)),change);

  %% PLOT Evolution of DENSITIES
  %colormap(gray); imagesc(-x); axis equal; axis tight; axis off;pause(1e-6);
end
colormap(gray); imagesc(-x); title(sprintf('Downward load at Node %.5g',loc)); axis equal; axis tight; axis off;pause(1e-6);drawnow;


end