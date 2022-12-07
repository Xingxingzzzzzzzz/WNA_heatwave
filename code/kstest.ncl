begin
 
   print("TXX7")

   f1 = addfile("/WORK2/zhangx/program/flow_ana/data2/CESM/cesm_hist_rcp85_txx7_eddyz500_ano.nc", "r")
   txx7_hist = f1->txx7(:,0:62)   ;1959-2021
   zxx7_hist = f1->eddyzxx7(:,0:62)
   printVarSummary(txx7_hist)
   printVarSummary(zxx7_hist(:,22:51))

   zxx7_hist_ano = zxx7_hist - conform(zxx7_hist,dim_avg_n(zxx7_hist(:,22:51), 1),0)
   txx7_hist_ano = txx7_hist - conform(txx7_hist,dim_avg_n(txx7_hist(:,22:51), 1),0)

   txx7_r  = f1->txx7(:,62:141)
   zxx7_r  = f1->eddyzxx7(:,62:141) ;2021-2100
   printVarSummary(txx7_r)

   txx7_r_ano = txx7_r - conform(txx7_r,dim_avg_n(txx7_hist(:,22:51),1),0)
   zxx7_r_ano = zxx7_r - conform(zxx7_r,dim_avg_n(zxx7_hist(:,22:51),1),0)

   f2 = addfile("/WORK2/zhangx/program/flow_ana/data2/CESM/cesm_picontrol_txx7_eddyz500_ano.nc", "r")
   txx7_pi  = f2->txx7
   zxx7_pi = f2->eddyzxx7

   zxx7_pi_ano = zxx7_pi - dim_avg_n_Wrap(dim_avg_n_Wrap(zxx7_hist(:,22:51),1),0)
   txx7_pi_ano = txx7_pi - dim_avg_n_Wrap(dim_avg_n_Wrap(txx7_hist(:,22:51),1),0)


   ;print(txx7_r_ano)

   f4             = addfile("/WORK2/zhangx/program/flow_ana/data2/ERA5/ERA5_txx7_eddyz500_ano.nc","r")
   txx7_era5      = f4->txx7
   txx7_era5_ano  = txx7_era5 - dim_avg(txx7_era5(22:51))


;;;;包含整段历史时期，为的是检验TXx7 anomaly的模拟(1959-2021)
   p1 = kolsm2_n(reshape(txx7_hist_ano,(/40*63/)), txx7_era5_ano,0) ;p1=0.2582978372939388,接受原假设,服从同一分布
   print(p1)

;;;;检验picontrol和历史时期的present-day period(1991-2020)模拟是否存在显著差异
   ;;;1.TXx7
   p1 = kolsm2_n(reshape(txx7_hist_ano(:,32:61),(/40*30/)), txx7_pi_ano,0) ;p1~0,拒绝原假设,不服从同一分布
   print(p1)

   ;;;;2.eddy z500
   p2 = kolsm2_n(reshape(zxx7_hist_ano(:,32:61),(/40*30/)), zxx7_pi_ano,0) ;p2=0.64,接受原假设，服从同一分布
   print(p2)

;;;;检验历史时期（present-day)和未来预估的pdf是否有显著差异
   ;;;;txx7
   p3 = kolsm2_n(reshape(txx7_r_ano(:,0:19),(/40*20/)), reshape(txx7_hist_ano(:,32:61),(/40*30/)),0) ;拒绝
   print(p3)
   ;;;p3~0,拒绝原假设，不服从同一分布

   ;;;zxx7
   p_z1 = kolsm2_n(reshape(zxx7_r_ano(:,0:19),(/40*20/)),reshape(zxx7_hist_ano(:,32:61),(/40*30/)),0)   ;拒绝
   print(p_z1)
   
   p_z2 = kolsm2_n(reshape(zxx7_r_ano(:,20:39),(/40*20/)),reshape(zxx7_hist_ano(:,32:61),(/40*30/)),0)   ;拒绝
   print(p_z2)

   p_z3 = kolsm2_n(reshape(zxx7_r_ano(:,60:79),(/40*20/)),reshape(zxx7_hist_ano(:,32:61),(/40*30/)),0)   ;拒绝
   print(p_z3) 
   ;;;均拒绝原假设，不服从同一分布



end
