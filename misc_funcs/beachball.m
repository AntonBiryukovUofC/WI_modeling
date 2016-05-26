% Program beachball
% D. Eaton
% 20110430


answer = inputdlg({'Strike (deg)','Dip (deg)','Rake (deg)'});
fm = str2num(char(answer));
fm = fm';

bb(fm,0,0,1,0,'r');



