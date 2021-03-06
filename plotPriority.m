% 20160317
% plot priority values
% merged_balloons_c3From1n5_log_0.5.txt

x = 1:1:54;
p50 = [0.9097	2.7226	0.0502	0.1244
0.9040	2.8010	0.0584	0.1479
0.9092	1.4677	0.0876	0.1169
0.8271	2.9108	0.0365	0.0878
0.9357	1.7500	0.0495	0.0810
0.8739	2.1108	0.0432	0.0796
0.8348	1.7422	0.0518	0.0754
0.9241	1.7579	0.0422	0.0685
0.8185	3.1069	0.0257	0.0654
0.8929	2.8481	0.0235	0.0598
0.9517	0.8167	0.0545	0.0424
0.9357	1.9932	0.0221	0.0412
0.8867	3.0206	0.0137	0.0368
0.9604	1.5618	0.0241	0.0362
0.8810	3.1932	0.0111	0.0312
0.7157	2.1579	0.0195	0.0301
0.9420	5.0990	0.0061	0.0291
0.7993	2.3853	0.0149	0.0285
0.8319	2.1814	0.0161	0.0292
0.9863	1.7265	0.0165	0.0281
0.8327	2.8481	0.0090	0.0212
0.9565	1.8834	0.0101	0.0182
0.9580	1.6324	0.0190	0.0297
0.8720	2.7383	0.0081	0.0193
0.7380	3.0912	0.0080	0.0182
0.8263	1.9618	0.0115	0.0187
0.8112	3.5618	0.0064	0.0186
0.7859	3.0990	0.0070	0.0170
0.9863	1.5853	0.0102	0.0160
0.7288	1.2211	0.0196	0.0175
0.8768	1.9069	0.0088	0.0147
0.8719	2.1814	0.0076	0.0144
0.6432	1.4374	0.0149	0.0137
0.7952	2.4167	0.0069	0.0133
0.9071	1.4598	0.0099	0.0131
0.8902	1.5932	0.0090	0.0127
0.7168	1.1069	0.0157	0.0124
0.9081	2.0918	0.0061	0.0115
0.8939	1.1775	0.0098	0.0103
0.8250	1.8990	0.0065	0.0102
0.9196	1.1989	0.0089	0.0098
0.8873	1.3764	0.0076	0.0093
0.5273	1.7535	0.0097	0.0090
0.6913	2.0559	0.0064	0.0091
0.7430	1.7422	0.0062	0.0081
0.5714	2.2912	0.0065	0.0085
0.8382	0.9030	0.0105	0.0079
0.6989	1.2010	0.0068	0.0057
0.6832	1.1853	0.0067	0.0054
0.6535	1.3343	0.0062	0.0054
0.8714	0.9892	0.0062	0.0054
0.7870	0.7461	0.0064	0.0037
0.9219	0.5892	0.0066	0.0036
0.7333	0.4559	0.0077	0.0026];

p25 = [0.934346	1.757863	0.020822	0.034199
0.951761	1.671588	0.020695	0.032925
0.986302	3.35002	0.008074	0.026678
0.953865	2.189235	0.011251	0.023494
0.986301	3.216686	0.006859	0.02176
0.720295	2.016686	0.015362	0.022316
0.9861	1.153941	0.014097	0.016041
0.925936	2.202736	0.008028	0.016373
0.941398	1.467667	0.00823	0.011372
0.944179	1.293203	0.009656	0.01179
0.986021	1.295118	0.008737	0.011158
0.980585	0.801	0.013339	0.010477
0.975309	0.411399	0.049327	0.019792
0.926664	1.373549	0.008159	0.010384
0.873205	0.887275	0.007448	0.005771];

p = semilogy(x,p50(:,1),x,p50(:,2),x,p50(:,3),x,p50(:,4));
xlim([1 54]);
ylim([0 5.1]);
xlabel('Iteration');
ylabel('Value (in log scale)');

ax = gca;
% ax.XTick = x;
grid on;

legend('Confidence','Data','Level','Priority',...
    'Location','east');

p(1).Marker = 'd';
p(1).LineStyle = '-';
p(1).Color = 'b';
p(1).MarkerFaceColor = 'b';

p(2).Marker = 'square';
p(2).LineStyle = ':';
p(2).Color = 'r';
p(2).MarkerFaceColor = 'r';

p(3).Marker = '^';
p(3).LineStyle = '-.';
p(3).Color = 'g';
p(3).MarkerFaceColor = 'g';

p(4).Marker = 'o';
p(4).LineStyle = '--';
p(4).Color = 'm';
p(4).MarkerFaceColor = 'm';

% 25 %

figure;

p = semilogy(x(1:15),p25(:,1),x(1:15),p25(:,2),...
    x(1:15),p25(:,3),x(1:15),p25(:,4));

xlim([1 15]);
ylim([0 5.1]);
xlabel('Iteration');
ylabel('Value (in log scale)');

ax = gca;
% ax.XTick = x;
grid on;

legend('Confidence','Data','Level','Priority',...
    'Location','east');

p(1).Marker = 'd';
p(1).LineStyle = '-';
p(1).Color = 'b';
p(1).MarkerFaceColor = 'b';

p(2).Marker = 'square';
p(2).LineStyle = ':';
p(2).Color = 'r';
p(2).MarkerFaceColor = 'r';

p(3).Marker = '^';
p(3).LineStyle = '-.';
p(3).Color = 'g';
p(3).MarkerFaceColor = 'g';

p(4).Marker = 'o';
p(4).LineStyle = '--';
p(4).Color = 'm';
p(4).MarkerFaceColor = 'm';


% no of holes

figure;

m50 = [421	798
415	782
410	772
404	754
393	742
389	726
382	716
371	696
366	686
356	668
352	656
349	644
345	625
341	613
339	603
332	590
314	580
310	568
298	549
287	529
286	521
276	511
273	499
271	493
264	478
250	459
244	443
235	425
225	406
224	398
214	384
208	368
204	356
184	329
171	309
167	298
160	278
145	261
139	245
133	228
125	204
120	190
115	168
83	166
76	147
65	138
55	115
53	103
43	79
30	61
14	42
11	30
2	16
1	8];

m25 = [57	192
51	173
48	161
47	153
44	141
43	133
26	114
25	106
21	92
18	74
15	62
14	54
13	46
12	41
8	23];

p = plot(x,m50(:,1),x,m50(:,2),x(1:15),m25(:,1),x(1:15),m25(:,2));
xlim([1 54]);
% ylim([0 421]);
xlabel('Iteration');
ylabel('No. of holes (pixels)');

ax = gca;
% ax.XTick = x;
grid on;

legend('No. of holes (50%)','Fill front count (50%)',...
    'No. of holes (25%)','Fill front count (25%)',...
    'Location','east');

p(1).Marker = '.';
p(1).LineStyle = '-';
p(1).Color = 'r';
p(1).MarkerFaceColor = 'r';

p(2).Marker = '+';
p(2).LineStyle = '-';
p(2).Color = 'r';
p(2).MarkerFaceColor = 'r';

p(3).Marker = '*';
p(3).LineStyle = ':';
p(3).Color = 'b';
p(3).MarkerFaceColor = 'b';

p(4).Marker = 'o';
p(4).LineStyle = ':';
p(4).Color = 'b';
p(4).MarkerFaceColor = 'b';

% p(3).Marker = '^';
% p(3).LineStyle = '-.';
% p(3).Color = 'g';
% p(3).MarkerFaceColor = 'g';
% 
% p(4).Marker = 'o';
% p(4).LineStyle = '--';
% p(4).Color = 'm';
% p(4).MarkerFaceColor = 'm';
% 
% p(5).Marker = 'x';
% p(5).LineStyle = '--';
% p(5).Color = 'k';
% p(5).MarkerFaceColor = 'k';