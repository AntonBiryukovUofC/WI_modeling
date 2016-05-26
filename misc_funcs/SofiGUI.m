function varargout = SofiGUI(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SofiGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @SofiGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
function addentry(handles,strings)
fl = fopen([pwd '/' 'logfile.txt'],'a+');
fprintf(fl,'%6s %6s \n',datestr(datetime('now')), strings);
fclose(fl);
Log_box_load(handles);
function Log_box_load(handles)
fl = fopen('logfile.txt','r');
CStr = textscan(fl,'%s', 'Delimiter', '\n');
set(handles.Log_box,'String',CStr{:},...
	'Value',numel(CStr{:}));
function SofiGUI_OpeningFcn(hObject, ~, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SofiGUI (see VARARGIN)
handles.rec_external=get(handles.checkbox_recfile,'Value');
handles.model_name = 'Model_Default' ;
handles.periodic_boundary = 0; % Turn Off Periodic Boundary by default.
handles.T_max = str2double(get(handles.edit_T,'String'));
handles.abframe = 15;
handles.IDYY =1;
handles.IDXX =1;
handles.IDZZ =1;
handles.shape=1;
handles.SETYPE = 1;
handles.TYPE=1;
handles.nsnaps =str2double(get(handles.edit_nsnaps,'String'));;
handles.snap=0; % Temporary;
handles.sources = [500 500 800];
handles.recb=[1 1 1];
handles.recend = [1500 1 1500 ];
handles.damping = 8.0;
handles.source_freq =1;
handles.Length_X = 2000;
handles.Length_Y = 2000;
handles.Length_Z = 2000;
handles.dh=str2double(get(handles.edit_dh,'String'));
handles.dt_mod=str2num(get(handles.edit_dt,'String'));
handles.recspan = str2double(get(handles.edit_XYstepRE,'String'));
handles.receivers = makeReceiverNet(handles);    
% Choose default command line output for SofiGUI
handles.output = hObject;
addpath('./misc_funcs');
try
h=fopen('./SofiGUI.txt');
if h==-1; 
	errordlg('Error opening SofiGUI.txt','Setup');
	return;
end 
n=1;
txt=1;
while txt~=-1
	txt=fgetl(h);
	if txt>-1	
		handles.run{n,1}=sscanf(txt,'%s:%s');
		n=n+1;
	end
end	
fclose(h);
catch me
    errordlg('Error opening SofiGUI.txt','Setup');
    return
end
handles.dir_data	= handles.run{1,1};
handles.exe_txt	= handles.run{2,1};
handles.exe_sof	= str2double(handles.run{3,1});
handles.exe_pdf = handles.run{4,1};  
handles.h=gcf;
%Open log file
addentry(handles,'This logfile tracks the SOFI3DGUI Errors. A new instance has launched. ');
addentry(handles,handles.exe_txt);
addentry(handles,handles.exe_sof);
addentry(handles,handles.exe_pdf);


% Update handles structure

guidata(hObject, handles);

% --------------------------------------------------------------------
function Open_Callback(hObject, ~, handles)
% hObject    handle to Open (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[handles.filename, handles.path]= uigetfile('*.mat','Select MAT file');
try
    set(handles.text_model,'string',[handles.path handles.filename]);
    handles1=load([handles.path handles.filename]);
catch me
    disp('Error Opening the model');
end
guidata(hObject,handles);


% --- Executes on button press in Generate.
function Generate_Callback(hObject, ~, handles)
% hObject    handle to Generate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%% create the JSON
handles = GenerateModelJSON(handles);

guidata(hObject,handles)

% --- Executes on key press with focus on Generate and none of its controls.
function Generate_KeyPressFcn(hObject, ~, handles)
% hObject    handle to Generate (see GCBO)
% eventdata  structure with the following fields (see UICONTROL)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)
guidata(hObject,handles)



function edit_T_Callback(hObject, ~, handles)
% hObject    handle to edit_T (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.T_max = str2double(get(hObject,'String'));
if isnan(handles.T_max)
    addentry(handles,' Wrong number at Simulation Time');
end
guidata(hObject,handles)



function edit_dt_Callback(hObject, ~, handles)
% hObject    handle to edit_dt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.dt_mod = str2num(get(hObject,'String'));
if isnan(handles.dt_mod)
    addentry(handles,' Wrong number at time step');
end
guidata(hObject,handles)



function edit_dh_Callback(hObject, eventdata, handles)
% hObject    handle to edit_dh (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.dh = str2num(get(hObject,'String'));
if isnan(handles.dh)
    addentry(handles,' Wrong number at time step');
end
guidata(hObject,handles)




function pushbutton_ani_Callback(hObject, ~, handles)
guidata(hObject,handles);
function Run_Callback(hObject, ~, handles)

RunTheModel(handles);

guidata(hObject,handles)
function Save_Callback(hObject, eventdata, handles)
mkdir('./results/');
save('./results/model_data.mat','handles');
gui_data=guidata(gcf);

save('./results/gui_data.mat','gui_data');
function edit_abstick_Callback(hObject, eventdata, handles)
handles.abframe = str2double(get(hObject,'String')) ;
guidata(hObject,handles)


function edit_nsnaps_Callback(hObject, eventdata, handles)
% hObject    handle to edit_nsnaps (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.nsnaps = str2double(get(hObject,'String')) ;
guidata(hObject,handles)

function SofiGUI_Callback(hObject, eventdata, handles)
unix([handles.exe_txt ' ./SofiGUI.txt &']);
function model_run_Callback(hObject, eventdata, handles)

unix([handles.exe_txt ' ./model_run.sh &']);
function modelrunjson_Callback(hObject, ~, handles)
unix([handles.exe_txt ' ./model_run.json &']);
function modelruntemplatejson_Callback(hObject, eventdata, handles)
unix([handles.exe_txt ' ./sofi3D_template.json&']);
function edit_PWAVED_Callback(hObject, eventdata, handles)

str= get(hObject,'String') ;
handles.src_single =str2double(strsplit(str,','));
guidata(hObject, handles);
update_3D_fig(handles);
if ~isnumeric(handles.src_single(1)) || ~isnumeric(handles.src_single(2)) || ~isnumeric(handles.src_single(3))
    addentry(handles,'Wrong single source coordinates !');
end


function Source_Timedelay_Callback(hObject, eventdata, handles)
% hObject    handle to Source_Timedelay (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
value = str2double(get(hObject,'String'));
if isnumeric(value)
    handles.lag=value;
else
    addentry(handles,'Wrong single source time lag');
end

function checkbox4_Callback(hObject, eventdata, handles)








% --- Executes on button press in checkbox_save.
function checkbox_save_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_save


% --- Executes on button press in checkbox_extpos.
function checkbox_extpos_Callback(hObject, eventdata, handles)
state_on=get(hObject,'Value');
if state_on
    handles.src_external=1;
    if exist('source.dat')>0
    source_params = ReadSourceFile();
    handles.sources = source_params(:,1:3);
    handles.source_freq = source_params(:,5);
    handles.amplitude = source_params(:,6);
    handles.lag = source_params(:,4);
    guidata(hObject,handles);
    else
        addentry(handles,'The file source.dat was not found in the current folder');
    end
else
    handles.src_external=0;
    handles.sources = MakeSource(handles);
    handles.source_freq = str2double(get(handles.Source_Freq,'String'));
    handles.amplitude = str2double(get(handles.Source_Amplitude,'String'));
    handles.lag =str2double(get(handles.Source_timedelay,'String')); 
     
    guidata(hObject,handles);
       
end

function sources = ReadSourceFile()
sources = load('./source.dat');

function sources = MakeSource(handles)
sources=handles.src_single;
        





% --- Executes on button press in checkbox_recfile.
function checkbox_recfile_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_recfile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
state_on=get(hObject,'Value');
if state_on
    handles.rec_external=1;
    if exist('receiver.dat')>0
    handles.receivers = ReadReceiverNet();
    guidata(hObject,handles);
    else
        addentry(handles,'The file receiver.dat was not found in the current folder');
    end
else
    handles.rec_external=0;
    handles.receivers = makeReceiverNet(handles);
    guidata(hObject,handles);

end
% Hint: get(hObject,'Value') returns toggle state of checkbox_recfile

function receivers = makeReceiverNet(handles)
[x,y,z]=meshgrid(handles.recb(1):handles.recspan:handles.recend(1),handles.recb(2):handles.recspan:handles.recend(2),handles.recb(3):handles.recspan:handles.recend(3));
receivers = [x(:) y(:) z(:)];

function receivers = ReadReceiverNet()
receivers = load('./receiver.dat');
    




function edit_XYZrec1_Callback(hObject, eventdata, handles)
% hObject    handle to edit_XYZrec1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_XYZrec1 as text
%        str2double(get(hObject,'String')) returns contents of edit_XYZrec1 as a double
str= get(hObject,'String') ;
recb =str2double(strsplit(str,','));
if ~isnumeric(recb)
    addentry(handles,'Wrong single receivers coordinates !');
    
end
handles.recb= recb;
if ~handles.rec_external
handles.receivers = makeReceiverNet(handles);
guidata(hObject,handles);
end



function edit_XYZrecEND_Callback(hObject, eventdata, handles)
% hObject    handle to edit_XYZrecEND (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str= get(hObject,'String') 
str2double(strsplit(str,','))
recend =str2double(strsplit(str,','));
if ~isnumeric(recend)
    addentry(handles,'Wrong single receivers coordinates !');
else    
    
    
handles.recend= recend;
if ~handles.rec_external
handles.receivers = makeReceiverNet(handles);
guidata(hObject,handles);
end


end



function edit_XYstepRE_Callback(hObject, eventdata, handles)
% hObject    handle to edit_XYstepRE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str= get(hObject,'String') ;
handles.recspan =str2double(str);
if ~handles.rec_external
handles.receivers = makeReceiverNet(handles);
guidata(hObject,handles);
end
if ~isnumeric(handles.recspan)
    addentry(handles,'Wrong single receivers span !');
end  

% Hints: get(hObject,'String') returns contents of edit_XYstepRE as text
%        str2double(get(hObject,'String')) returns contents of edit_XYstepRE as a double



% --- Executes on button press in Close_figs.
function Close_figs_Callback(hObject, eventdata, handles)
% hObject    handle to Close_figs (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
figHandles = findobj('Type','figure');
close(figHandles(figHandles~=gcf))



% --- Executes on button press in pushbutton_genMAT.
function pushbutton_genMAT_Callback(hObject, eventdata, handles)

guidata(hObject,handles);


% --------------------------------------------------------------------


% --- Executes on button press in start_xterm.
function start_xterm_Callback(hObject, eventdata, handles)
% hObject    handle to start_xterm (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
unix('xterm &');

function update_graphs(handles)
% hObject    handle to start_xterm (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
cols_def = get(handles.vpvs_rho_fig,'ColorOrder')
%cols = hsv(numel(handles.Depths));
if size(cols_def,1)>numel(handles.Depths)
cols = cols_def;
handles.cols = cols;
else
handles.cols = hsv(numel(handles.Depths));
end

axes(handles.vpvs_rho_fig);
cla;
 %keyboard
 Y=[0; handles.Depths];
 XP=[handles.Vp(1); handles.Vp];
 XS=[handles.Vs(1); handles.Vs];
 stairs(XP,Y,'-k','LineWidth',2);
 hold all;
 stairs(XS,Y,'-k','LineWidth',2);
  for i=2:numel(Y)
 stairs([XP(i-1) XP(i)],[Y(i-1) Y(i)],'LineStyle','--','LineWidth',2,'Color',cols(i-1,:));
 stairs([XS(i-1) XS(i)],[Y(i-1) Y(i)],'LineStyle','--','LineWidth',2,'Color',cols(i-1,:));
 end
  set(handles.vpvs_rho_fig,'YDir','reverse');
  xlim([0.8*min(XS) 1.1*max(XP)]);
  ylim([0 max(Y)*1.1]);

 %-------------------------------------------__%
 
  axes(handles.qp_qs_fig);
  cla;
 Y=[0; handles.Depths];
 XR=[handles.Rho(1); handles.Rho];
 stairs(XR,Y,'-k','LineWidth',2);
 hold all
 for i=2:numel(Y)
 stairs([XR(i-1) XR(i)],[Y(i-1) Y(i)],'LineStyle','--','LineWidth',2,'Color',cols(i-1,:));
 end
 set(handles.qp_qs_fig,'YDir','reverse');
 xlim([0.8*min(XR) 1.1*max(XR)]);
 ylim([0 max(Y)*1.1]);
 update_3D_fig(handles);
 
 
 function update_3D_fig(handles)
     axes(handles.geosection_plot)
     cla;

      
     %n=20;
     %dh = [handles.Length_X handles.Length_Y handles.Length_Z]/n;
     dh =[ handles.dh handles.dh handles.dh];
     
     [X,Y,Z] = meshgrid(0:dh(1):handles.Length_X,0:dh(2):handles.Length_Y,0:dh(3):handles.Length_Z);
     Vp3d=0.*Z;
     for i=numel(handles.Depths):-1:1
     Vp3d(Z<=handles.Depths(i)) = handles.Vp(i);
     end
     xslice = X(10,10); 
    zslice = handles.sources(:,3); 
    yslice =Y(10,10)
    
    sl = slice(X,Y,Z,Vp3d,xslice,yslice,zslice);
    
    
    hold all;
    %ylim([handles.rec1(2) handles.recend(2)])
    %zlim([0 handles.Depths(end)])
    %xlim([handles.rec1(1) handles.recend(1)])
    %handles.src=handles.src_single; % TEMPORARY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    plot3(handles.sources(:,1),handles.sources(:,2),handles.sources(:,3),'.k','markersize',60,'MarkerFaceColor','white');
    set(handles.geosection_plot,'ZDir','reverse');
    %set(sl,'LineStyle','none');
    cameratoolbar;
    %line([0+handles.abframe*dh
   nx =floor(handles.Length_X/handles.dh);
    ny = floor(handles.Length_Y/handles.dh);
    nz = floor(handles.Length_Z/handles.dh);
    addentry(handles,['The size of the model is : ' num2str(nx) '-x '  num2str(ny) '-y' num2str(nz) '-z']);
     
function update_receivers(handles)
    axes(handles.geosection_plot)
    cla;
    update_3D_fig(handles);
    plot3(handles.receivers(:,1),handles.receivers(:,2),handles.receivers(:,3),'vk','markersize',30,'MarkerFaceColor','g');







% --- Executes on button press in ImportVPVS.
function ImportVPVS_Callback(hObject, eventdata, handles)
% hObject    handle to ImportVPVS (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[handles.filenameVpVs, handles.pathVpVs]= uigetfile('VpVs.dat','Select DAT file');
[handles.Depths, handles.Vp, handles.Vs, handles.Rho, handles.Qp, handles.Qs]=VpVsImport([handles.pathVpVs handles.filenameVpVs]);

 tableData = cell(numel(handles.Depths),7);
 cols = hsv(numel(handles.Depths));
 for i=1: numel(handles.Depths)
 color = rgb2hex(cols(i,:));    
 tableData{i,1}=['<html><body bgcolor="' color '">Layer</body></html>'];
 tableData(i,2:end) = {handles.Depths(i), handles.Vp(i), handles.Vs(i), handles.Rho(i), handles.Qp(i), handles.Qs(i)};
 end
 %keyboard
 set(handles.VelocityRhoModel, 'data',tableData);
 addentry(handles,'Geology model successfully opened');
 update_graphs(handles)
    
%addentry(handles,'Error opening VpVs models');
guidata(hObject,handles);

 %uitable('Data',{'<html><body bgcolor="#FF0000">Hello</body></html>'})
 

% --- Executes when entered data in editable cell(s) in VelocityRhoModel.
function VelocityRhoModel_CellEditCallback(hObject, eventdata, handles)
% hObject    handle to VelocityRhoModel (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.TABLE)
%	Indices: row and column indices of the cell(s) edited
%	PreviousData: previous data for the cell(s) edited
%	EditData: string(s) entered by the user
%	NewData: EditData or its converted form set on the Data property. Empty if Data was not changed
%	Error: error string when failed to convert EditData to appropriate value for Data
% handles    structure with handles and user data (see GUIDATA)
row = eventdata.Indices(1);
column= eventdata.Indices(2);
switch column
    case 2
        handles.Depths(row)=str2double(eventdata.EditData);
    %keyboard
    case 3
        handles.Vp(row)=str2double(eventdata.EditData);
    case 4
        handles.Vs(row)=str2double(eventdata.EditData);
    case 5
        handles.Rho(row)=str2double(eventdata.EditData);
    case 6
        handles.Qp(row)=str2double(eventdata.EditData);
    case 7
        handles.Qs(row)=str2double(eventdata.EditData);
end 
guidata(hObject,handles);

update_graphs(handles)
addentry(handles,'Property was changed, graphs updated');        




% --- Executes when selected cell(s) is changed in VelocityRhoModel.
function VelocityRhoModel_CellSelectionCallback(hObject, eventdata, handles)


% --- Executes on button press in ShowGeometry.
function ShowGeometry_Callback(hObject, eventdata, handles)
% hObject    handle to ShowGeometry (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
update_receivers(handles);



function Source_Freq_Callback(hObject, ~, handles)
% hObject    handle to Source_Freq (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
value = str2double(get(hObject,'String'));
if isnumeric(value) && (~handles.src_external)
    handles.source_freq =value;
else
    addentry(handles,'Wrong single source frequency');
end
guidata(hObject,handles);


% Hints: get(hObject,'String') returns contents of Source_Freq as text
%        str2double(get(hObject,'String')) returns contents of Source_Freq as a double



% --- Executes on button press in Dh_Dt_automatic.
function Dh_Dt_automatic_Callback(hObject, ~, handles)
% hObject    handle to Dh_Dt_automatic (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
state_on=get(hObject,'Value');
if state_on
    handles.Automatic=1;
    handles = reset_dt_dh(handles);
    set(handles.edit_dt,'String',num2str(handles.dt_mod));
    set(handles.edit_dh,'String',num2str(handles.dh));
    addentry(handles,['The values of DT and DH were reset to ' num2str(handles.dt_mod) ' and ' num2str(handles.dh) ]);
    guidata(hObject,handles);
else
    handles.Automatic=0;
    handles.dh = str2double(get(handles.edit_dh,'String'));
    handles.dt = str2double(get(handles.edit_dt,'String'));
     guidata(hObject,handles);
    
end
% Hint: get(hObject,'Value') returns toggle state of Dh_Dt_automatic
    
    function handles_new=reset_dt_dh(handles)
    n=13;
    rmax = 0.587;
    
    min_lambda = min(handles.Vs)/handles.source_freq;
    handles.dh =min_lambda/(2*n);
    handles.dt_mod = handles.dh/(rmax*max(handles.Vp));
    handles_new=handles;

    
    
    
        
        
        

% --- Executes on button press in ShowModel.
function ShowModel_Callback(hObject, ~, handles)
% hObject    handle to ShowModel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in LoadModel.
function LoadModel_Callback(hObject, eventdata, handles)
% hObject    handle to LoadModel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[handles.filename handles.path]= uigetfile('gui_data.mat','Select MAT file');
try
    gui_data=load([handles.path handles.filename]);
catch me
    disp('Error(3)');
end



% --- Executes on button press in ShowRadiationPattern.
function ShowRadiationPattern_Callback(hObject, eventdata, handles)
% hObject    handle to ShowRadiationPattern (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in DIsplay_Model.
function DIsplay_Model_Callback(hObject, eventdata, handles)
% hObject    handle to DIsplay_Model (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



% --- Executes on button press in expl_source.
function expl_source_Callback(hObject, eventdata, handles)
% hObject    handle to expl_source (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
state_on=get(hObject,'Value');
if state_on
    handles.TYPE=1;
end
guidata(hObject,handles);


% --- Executes on button press in forceXtype.
function forceXtype_Callback(hObject, eventdata, handles)
% hObject    handle to forceXtype (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
state_on=get(hObject,'Value');
if state_on
    handles.TYPE=2;
end
guidata(hObject,handles);


% --- Executes on button press in forceYtype.
function forceYtype_Callback(hObject, eventdata, handles)
% hObject    handle to forceYtype (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
state_on=get(hObject,'Value');
if state_on
    handles.TYPE=3;
end
guidata(hObject,handles);

% Hint: get(hObject,'Value') returns toggle state of forceYtype


% --- Executes on button press in forceZtype.
function forceZtype_Callback(hObject, eventdata, handles)
% hObject    handle to forceZtype (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
state_on=get(hObject,'Value');
if state_on
    handles.TYPE=4;
end
guidata(hObject,handles);

% Hint: get(hObject,'Value') returns toggle state of forceZtype


% --- Executes on button press in ricker.
function ricker_Callback(hObject, eventdata, handles)
% hObject    handle to ricker (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
state_on=get(hObject,'Value');
if state_on
    handles.shape=1;
end
guidata(hObject,handles);

% Hint: get(hObject,'Value') returns toggle state of ricker


% --- Executes on button press in fumue.
function fumue_Callback(hObject, eventdata, handles)
% hObject    handle to fumue (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
state_on=get(hObject,'Value');
if state_on
    handles.shape=2;
end
guidata(hObject,handles);
% Hint: get(hObject,'Value') returns toggle state of fumue


% --- Executes on button press in shape_file.
function shape_file_Callback(hObject, eventdata, handles)
% hObject    handle to shape_file (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
state_on=get(hObject,'Value');
if state_on
    handles.shape=3;
end
guidata(hObject,handles);
% Hint: get(hObject,'Value') returns toggle state of shape_file


% --- Executes on button press in sin3.
function sin3_Callback(hObject, eventdata, handles)
% hObject    handle to sin3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
state_on=get(hObject,'Value');
if state_on
    handles.shape=4;
end
guidata(hObject,handles);
% Hint: get(hObject,'Value') returns toggle state of sin3






%_____________________________%_______________________________________________________________________________________________________________________%


function edit_dh_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function Generate_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function edit_T_CreateFcn(hObject, ~, handles)

guidata(hObject,handles)
function edit_dt_CreateFcn(hObject, eventdata, handles)

guidata(hObject,handles)
function pushbutton_ani_CreateFcn(hObject, eventdata, handles)

guidata(hObject,handles)
function Run_CreateFcn(hObject, eventdata, handles)
guidata(hObject,handles)
function Save_CreateFcn(hObject, eventdata, handles)
 
guidata(hObject,handles)
function edit_abstick_CreateFcn(hObject, eventdata, handles)
guidata(hObject,handles)
function edit_nsnaps_CreateFcn(hObject, eventdata, handles)
guidata(hObject,handles)
function SofiGUI_CreateFcn(hObject, eventdata, handles)
guidata(hObject,handles)
function model_run_CreateFcn(hObject, eventdata, handles)

guidata(hObject,handles)
function modelrunjson_CreateFcn(hObject, eventdata, handles)
guidata(hObject,handles)
function modelruntemplatejson_CreateFcn(hObject, eventdata, handles)
guidata(hObject,handles)
function edit_PWAVED_CreateFcn(hObject, eventdata, handles)
guidata(hObject,handles)
function checkbox_extpos_CreateFcn(hObject, eventdata, handles)
guidata(hObject,handles)
function Source_Timedelay_CreateFcn(hObject, eventdata, handles)

guidata(hObject,handles)
function sourcedat_CreateFcn(hObject, eventdata, handles)
guidata(hObject,handles)
function receiverdat_CreateFcn(hObject, eventdata, handles)
guidata(hObject,handles)
function sofi2dpdf_CreateFcn(hObject, eventdata, handles)
guidata(hObject,handles)
function checkbox4_CreateFcn(hObject, eventdata, handles)
guidata(hObject,handles)
function checkbox_save_CreateFcn(hObject, eventdata, handles)
guidata(hObject,handles)
function checkbox_recfile_CreateFcn(hObject, eventdata, handles)
guidata(hObject,handles)
function edit_XYZrec1_CreateFcn(hObject, eventdata, handles)
guidata(hObject,handles)
function edit_XYZrecEND_CreateFcn(hObject, eventdata, handles)
guidata(hObject,handles)
function edit_XYstepRE_CreateFcn(hObject, eventdata, handles)
guidata(hObject,handles)
function Close_figs_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function edit_FLL_CreateFcn(hObject, ~, ~)
function pushbutton_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function start_xterm_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function ImportVPVS_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function ShowGeometry_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function Source_Freq_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function Dh_Dt_automatic_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function ShowModel_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function LoadModel_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function ShowRadiationPattern_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function DIsplay_Model_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function VelocityRhoModel_CreateFcn(hObject, ~, handles)

%__________________________________________________________________________________________________________________________________________________________%
function expl_source_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function Length_Z_CreateFcn(hObject, eventdata, handles)
% --- Executes during object creation, after setting all properties.
function Length_X_CreateFcn(hObject, eventdata, handles)
function ricker_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function forceYtype_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function forceZtype_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function forceXtype_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function fumue_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function sin3_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function shape_file_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
function edit_TTAU_CreateFcn(hObject, ~, handles)
guidata(hObject,handles)
% --- Executes during object creation, after setting all properties.
function text_model_CreateFcn(~, ~, ~)
function Source_Amplitude_CreateFcn(hObject, eventdata, handles)

function figure1_CreateFcn(~, ~, ~)
function forceXtype_DeleteFcn(~, ~, ~)
function File_Callback(~, ~, ~)
function Scripts_Callback(~, ~, ~)
function Menu_Callback(~, ~, ~)
function help_Callback(~, ~, ~)
function expl_source_DeleteFcn(~, ~, ~)
function Log_box_CreateFcn(hObject, eventdata, handles)
function Log_box_Callback(~, ~, ~)
function sourcedat_Callback(hObject, eventdata, handles)

unix([handles.exe_txt ' ./source.dat &']);
function receiverdat_Callback(hObject, eventdata, handles)

unix([handles.exe_txt ' ./receiver.dat &']);
function sofi2dpdf_Callback(hObject, eventdata, handles)
unix([handles.exe_pdf ' ./guide_sofi3D.pdf &']);

% --- Outputs from this function are returned to the command line.
function varargout = SofiGUI_OutputFcn(~, ~, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
%varargout{1} = handles.output;
% --------------------------------------------------------------------


% --- Executes on key press with focus on Log_box and none of its controls.
function Log_box_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to Log_box (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.UICONTROL)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pressure_seismo.
function pressure_seismo_Callback(hObject, eventdata, handles)
% hObject    handle to pressure_seismo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
fl = get(hObject,'Value');
if fl 
    handles.SETYPE = 2;
end
guidata(hObject,handles)

% Hint: get(hObject,'Value') returns toggle state of pressure_seismo


% --- Executes on button press in velocity_seismo.
function velocity_seismo_Callback(hObject, eventdata, handles)
% hObject    handle to velocity_seismo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
fl = get(hObject,'Value');
if fl 
    handles.SETYPE = 1;
end
guidata(hObject,handles)

% Hint: get(hObject,'Value') returns toggle state of velocity_seismo


% --- Executes on button press in all_seismo.
function all_seismo_Callback(hObject, eventdata, handles)
% hObject    handle to all_seismo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
fl = get(hObject,'Value');
if fl 
    handles.SETYPE = 4;
end
guidata(hObject,handles)

% Hint: get(hObject,'Value') returns toggle state of all_seismo



function Source_Amplitude_Callback(hObject, eventdata, handles)
% hObject    handle to Source_Amplitude (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
value = str2double(get(hObject,'String'));
if isnumeric(value)
    handles.amplitude=value;
else
    addentry(handles,'Wrong single source amplitude');
end
guidata(hObject,handles)

% Hints: get(hObject,'String') returns contents of Source_Amplitude as text
%        str2double(get(hObject,'String')) returns contents of Source_Amplitude as a double


% --- Executes during object creation, after setting all properties.



function Length_Z_Callback(hObject, eventdata, handles)
value = str2double(get(hObject,'String'));
if isnumeric(value)
    handles.Length_Z=value;
else
    addentry(handles,'Wrong Length in Z');
end
guidata(hObject,handles)
update_receivers(handles)



% --- Executes during object creation, after setting all properties.


function Length_Y_Callback(hObject, eventdata, handles)
value = str2double(get(hObject,'String'));
if isnumeric(value)
    handles.Length_Y=value;
else
    addentry(handles,'Wrong Length in Y');
end
guidata(hObject,handles)
update_receivers(handles)


% --- Executes during object creation, after setting all properties.
function Length_Y_CreateFcn(hObject, eventdata, handles)




function Length_X_Callback(hObject, eventdata, handles)
value = str2double(get(hObject,'String'));
if isnumeric(value)
    handles.Length_X=value;
else
    addentry(handles,'Wrong Length in X');
end
guidata(hObject,handles)
update_receivers(handles)




function model_name_Callback(hObject, eventdata, handles)
handles.model_name = get(hObject,'String') ;


% --- Executes during object creation, after setting all properties.
function model_name_CreateFcn(hObject, eventdata, handles)
