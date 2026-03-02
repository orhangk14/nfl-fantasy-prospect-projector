# data_collection/parse_2026_combine.py
"""
Parse the 2026 combine data from Tankathon into a clean CSV.
Only saves fantasy-relevant positions (QB, RB, WR, TE) but 
parses all for completeness.

Run: python -m data_collection.parse_2026_combine
"""

import csv
import os
import re

OUTPUT_DIR = "data/raw"

RAW_DATA = """1	Ohio State	Arvell Reese	LB	6'4"	241 lbs	40-Yard	4.46	Vertical		Broad		Shuttle		3-Cone	
2	Miami	Rueben Bain Jr.	EDGE	6'2.5"	263 lbs	40-Yard		Vertical		Broad		Bench		3-Cone	
3	Ohio State	Caleb Downs	S	5'11.5"	206 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
4	Indiana	Fernando Mendoza	QB	6'5"	236 lbs	40-Yard		Vertical		Broad		Shuttle		Hand	9.5"
5	Texas Tech	David Bailey	EDGE	6'3.5"	251 lbs	40-Yard	4.50	Vertical	35.0"	Broad	129"	Bench		3-Cone	
6	Miami	Francis Mauigoa	OT	6'5.5"	329 lbs	40-Yard		Vertical		Broad		Bench		Arm	33.25"
7	Ohio State	Sonny Styles	LB	6'5"	244 lbs	40-Yard	4.46	Vertical	43.5"	Broad	134"	Shuttle	4.26	3-Cone	7.09
8	Utah	Spencer Fano	OT	6'5.5"	311 lbs	40-Yard	4.91	Vertical	32.0"	Broad	111"	Bench		Arm	32.125"
9	Ohio State	Carnell Tate	WR	6'2.5"	192 lbs	40-Yard	4.53	Vertical		Broad		Shuttle		3-Cone	
10	Notre Dame	Jeremiyah Love	RB	6'0"	212 lbs	40-Yard	4.36	Vertical		Broad		Shuttle		3-Cone	
11	LSU	Mansoor Delane	CB	6'0"	187 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
12	USC	Makai Lemon	WR	5'11"	192 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
13	Auburn	Keldric Faulk	EDGE	6'6"	276 lbs	40-Yard		Vertical	35.0"	Broad	117"	Bench		3-Cone	
14	Oregon	Kenyon Sadiq	TE	6'3"	241 lbs	40-Yard	4.39	Vertical	43.5"	Broad	133"	Shuttle		3-Cone	
15	Arizona State	Jordyn Tyson	WR	6'2"	203 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
16	Penn State	Vega Ioane	IOL	6'4.5"	320 lbs	40-Yard		Vertical	31.5"	Broad	104"	Bench		Arm	32.75"
17	Clemson	Peter Woods	DL	6'2.5"	298 lbs	40-Yard		Vertical		Broad		Bench		3-Cone	
18	Washington	Denzel Boston	WR	6'3.5"	212 lbs	40-Yard		Vertical	35.0"	Broad		Shuttle	4.28	3-Cone	
19	Georgia	Monroe Freeling	OT	6'7.5"	315 lbs	40-Yard	4.93	Vertical	33.5"	Broad	115"	Bench		Arm	34.75"
20	Utah	Caleb Lomu	OT	6'6.5"	313 lbs	40-Yard	4.99	Vertical	32.5"	Broad	113"	Bench		Arm	33.375"
21	Tennessee	Jermod McCoy	CB	6'1"	188 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
22	Texas A&M	Cashius Howell	EDGE	6'2.5"	253 lbs	40-Yard	4.59	Vertical	32.5"	Broad	115"	Bench		3-Cone	
23	Clemson	Avieon Terrell	CB	5'11"	186 lbs	40-Yard		Vertical	34.0"	Broad	123"	Shuttle		3-Cone	
24	Alabama	Kadyn Proctor	OT	6'6.5"	352 lbs	40-Yard	5.21	Vertical	32.5"	Broad	109"	Bench		Arm	33.375"
25	Miami	Akheem Mesidor	EDGE	6'3"	259 lbs	40-Yard		Vertical		Broad		Bench		3-Cone	
26	Texas A&M	KC Concepcion	WR	5'11.5"	196 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
27	Oregon	Dillon Thieneman	S	6'0"	201 lbs	40-Yard	4.35	Vertical	41.0"	Broad	125"	Shuttle		3-Cone	
28	Florida	Caleb Banks	DL	6'6.5"	327 lbs	40-Yard	5.04	Vertical	32.0"	Broad	114"	Bench		3-Cone	
29	Georgia	CJ Allen	LB	6'1"	230 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
30	Ohio State	Kayden McDonald	DL	6'2"	326 lbs	40-Yard		Vertical		Broad		Bench		3-Cone	
31	Alabama	Ty Simpson	QB	6'1"	211 lbs	40-Yard		Vertical		Broad		Shuttle		Hand	9.375"
32	Toledo	Emmanuel McNeil-Warren	S	6'3.5"	201 lbs	40-Yard	4.52	Vertical	35.5"	Broad	122"	Shuttle		3-Cone	
33	Clemson	T.J. Parker	EDGE	6'3.5"	263 lbs	40-Yard	4.68	Vertical	34.0"	Broad	120"	Bench		3-Cone	
34	South Carolina	Brandon Cisse	CB	6'0"	189 lbs	40-Yard		Vertical	41.0"	Broad	131"	Shuttle		3-Cone	
35	Tennessee	Colton Hood	CB	5'11.5"	193 lbs	40-Yard	4.44	Vertical	40.5"	Broad	125"	Shuttle		3-Cone	
36	Texas Tech	Lee Hunter	DL	6'3.5"	318 lbs	40-Yard	5.18	Vertical	21.5"	Broad	100"	Bench		3-Cone	
37	Clemson	Blake Miller	OT	6'7"	317 lbs	40-Yard	5.04	Vertical	32.0"	Broad	113"	Bench		Arm	34.25"
38	Arizona State	Max Iheanachor	OT	6'6"	321 lbs	40-Yard	4.91	Vertical	30.5"	Broad	115"	Bench		Arm	33.875"
39	Indiana	Omar Cooper Jr.	WR	6'0"	199 lbs	40-Yard	4.42	Vertical	37.0"	Broad		Shuttle		3-Cone	
40	Missouri	Zion Young	EDGE	6'6"	262 lbs	40-Yard		Vertical		Broad		Bench		3-Cone	
41	Oregon	Emmanuel Pregnon	IOL	6'4.5"	314 lbs	40-Yard	5.21	Vertical	35.0"	Broad	111"	Bench		Arm	33.625"
42	Oklahoma	R Mason Thomas	EDGE	6'2.5"	241 lbs	40-Yard	4.67	Vertical		Broad		Bench		3-Cone	
43	Texas	Anthony Hill Jr.	LB	6'2"	238 lbs	40-Yard	4.51	Vertical	37.0"	Broad	125"	Shuttle		3-Cone	
44	Indiana	D'Angelo Ponds	CB	5'8.5"	182 lbs	40-Yard		Vertical	43.5"	Broad		Shuttle		3-Cone	
45	Notre Dame	Malachi Fields	WR	6'4.5"	218 lbs	40-Yard	4.61	Vertical	38.0"	Broad	124"	Shuttle	4.35	3-Cone	6.98
46	Georgia	Christen Miller	DL	6'4"	321 lbs	40-Yard		Vertical		Broad		Bench		3-Cone	
47	San Diego State	Chris Johnson	CB	6'0.5"	193 lbs	40-Yard	4.40	Vertical	38.0"	Broad	126"	Shuttle		3-Cone	
48	Louisville	Chris Bell	WR	6'2"	222 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
49	LSU	A.J. Haulcy	S	5'11.5"	215 lbs	40-Yard	4.52	Vertical		Broad		Shuttle		3-Cone	
50	Georgia	Zachariah Branch	WR	5'8.5"	177 lbs	40-Yard	4.35	Vertical	38.0"	Broad	125"	Shuttle		3-Cone	
51	Notre Dame	Jadarian Price	RB	5'10.5"	203 lbs	40-Yard	4.49	Vertical	35.0"	Broad	124"	Shuttle		3-Cone	
52	Tennessee	Chris Brazzell II	WR	6'4"	198 lbs	40-Yard	4.37	Vertical		Broad		Shuttle		3-Cone	
53	Texas A&M	Chase Bisontis	IOL	6'5.5"	315 lbs	40-Yard	5.02	Vertical	32.0"	Broad	105"	Bench		Arm	31.75"
54	Illinois	Gabe Jacas	EDGE	6'3.5"	260 lbs	40-Yard		Vertical		Broad		Bench	30	3-Cone	
55	Arizona State	Keith Abney II	CB	5'10"	187 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
56	Alabama	Germie Bernard	WR	6'1.5"	206 lbs	40-Yard	4.48	Vertical	32.5"	Broad	125"	Shuttle	4.31	3-Cone	6.71
57	Cincinnati	Jake Golday	LB	6'4.5"	239 lbs	40-Yard	4.62	Vertical	39.0"	Broad	125"	Shuttle	4.34	3-Cone	7.02
58	Miami	Keionte Scott	CB	5'11.5"	193 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
59	Northwestern	Caleb Tiernan	OT	6'7.5"	323 lbs	40-Yard		Vertical	35.5"	Broad	111"	Bench		Arm	32.25"
60	Alabama	LT Overton	EDGE	6'3"	274 lbs	40-Yard	4.87	Vertical		Broad		Bench		3-Cone	
61	Iowa	Gennings Dunker	OT	6'5"	319 lbs	40-Yard	5.18	Vertical	32.5"	Broad	108"	Bench		Arm	33.5"
62	Indiana	Elijah Sarratt	WR	6'2.5"	210 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
63	Missouri	Josiah Trotter	LB	6'2"	237 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
64	Vanderbilt	Eli Stowers	TE	6'4"	239 lbs	40-Yard	4.51	Vertical	45.5"	Broad	135"	Shuttle		3-Cone	
65	Michigan	Derrick Moore	EDGE	6'4"	255 lbs	40-Yard		Vertical		Broad		Bench		3-Cone	
66	Auburn	Connor Lew	IOL	6'3.5"	310 lbs	40-Yard		Vertical		Broad		Bench		Arm	32.375"
67	USC	Kamari Ramsey	S	6'0.5"	202 lbs	40-Yard	4.47	Vertical	36.0"	Broad	120"	Shuttle		3-Cone	
68	Clemson	Antonio Williams	WR	5'11.5"	187 lbs	40-Yard	4.41	Vertical	39.5"	Broad	124"	Shuttle		3-Cone	7.00
69	Texas Tech	Jacob Rodriguez	LB	6'1.5"	231 lbs	40-Yard	4.57	Vertical	38.5"	Broad	121"	Shuttle	4.19	3-Cone	6.90
70	Ohio State	Davison Igbinosun	CB	6'2"	189 lbs	40-Yard	4.45	Vertical	34.0"	Broad	120"	Shuttle		3-Cone	
71	Texas Tech	Romello Height	EDGE	6'3"	239 lbs	40-Yard	4.64	Vertical	39.0"	Broad	125"	Bench		3-Cone	
72	Ohio State	Max Klare	TE	6'4.5"	246 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
73	Iowa State	Domonique Orange	DL	6'2.5"	322 lbs	40-Yard		Vertical		Broad		Bench		3-Cone	
74	Pittsburgh	Kyle Louis	LB	6'0"	220 lbs	40-Yard	4.53	Vertical	39.5"	Broad	129"	Shuttle	4.26	3-Cone	6.97
75	Nebraska	Emmett Johnson	RB	5'10.5"	202 lbs	40-Yard	4.56	Vertical	35.5"	Broad	120"	Shuttle	4.29	3-Cone	7.32
76	Tennessee	Joshua Josephs	EDGE	6'3"	242 lbs	40-Yard		Vertical		Broad		Bench		3-Cone	
77	Alabama	Deontae Lawson	LB	6'3"	226 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
78	Texas	Malik Muhammad	CB	6'0"	182 lbs	40-Yard	4.42	Vertical	39.0"	Broad	130"	Shuttle		3-Cone	
79	Florida State	Darrell Jackson Jr.	DL	6'5.5"	315 lbs	40-Yard		Vertical		Broad		Bench		3-Cone	
80	Arkansas	Julian Neal	CB	6'1.5"	203 lbs	40-Yard	4.49	Vertical	40.0"	Broad	134"	Shuttle	4.20	3-Cone	7.13
81	Washington	Jonah Coleman	RB	5'8"	220 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
82	Florida	Devin Moore	CB	6'3.5"	198 lbs	40-Yard	4.50	Vertical		Broad		Shuttle		3-Cone	
83	LSU	Garrett Nussmeier	QB	6'1.5"	203 lbs	40-Yard		Vertical		Broad		Shuttle		Hand	9.125"
84	UCF	Malachi Lawrence	EDGE	6'4.5"	253 lbs	40-Yard	4.52	Vertical	40.0"	Broad	130"	Bench		3-Cone	
85	Penn State	Dani Dennis-Sutton	EDGE	6'5.5"	256 lbs	40-Yard	4.63	Vertical	39.5"	Broad	131"	Bench		3-Cone	6.90
86	Baylor	Michael Trigg	TE	6'4"	240 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
87	Arizona	Genesis Smith	S	6'2"	202 lbs	40-Yard		Vertical	42.5"	Broad	128"	Shuttle	4.18	3-Cone	
88	Duke	Chandler Rivers	CB	5'9.5"	185 lbs	40-Yard	4.40	Vertical	39.0"	Broad	130"	Shuttle		3-Cone	
89	Arkansas	Mike Washington Jr.	RB	6'1"	223 lbs	40-Yard	4.33	Vertical	39.0"	Broad	128"	Shuttle		3-Cone	
90	USC	Ja'Kobi Lane	WR	6'4.5"	200 lbs	40-Yard	4.47	Vertical	40.0"	Broad	129"	Shuttle		3-Cone	
91	Arizona	Treydan Stukes	CB	6'0.5"	190 lbs	40-Yard	4.33	Vertical	38.0"	Broad	130"	Shuttle		3-Cone	
92	Georgia State	Ted Hurst	WR	6'4"	206 lbs	40-Yard	4.42	Vertical	36.5"	Broad	135"	Shuttle		3-Cone	
93	Oklahoma	Deion Burks	WR	5'10"	180 lbs	40-Yard	4.30	Vertical	42.5"	Broad	131"	Shuttle		3-Cone	
94	Penn State	Zakee Wheatley	S	6'3"	203 lbs	40-Yard		Vertical	32.5"	Broad	122"	Shuttle		3-Cone	
95	USC	Anthony Lucas	EDGE	6'5.5"	256 lbs	40-Yard		Vertical		Broad		Bench		3-Cone	
96	Texas A&M	Will Lee III	CB	6'1.5"	189 lbs	40-Yard	4.52	Vertical	42.0"	Broad	132"	Shuttle		3-Cone	
97	Georgia Tech	Keylan Rutledge	IOL	6'3.5"	316 lbs	40-Yard	5.05	Vertical	32.5"	Broad	104"	Bench		Arm	33.25"
98	Kansas State	Sam Hecht	IOL	6'4"	303 lbs	40-Yard	5.10	Vertical	28.0"	Broad	101"	Bench		Arm	31.625"
99	Missouri	Chris McClellan	DL	6'4"	313 lbs	40-Yard	5.05	Vertical	29.5"	Broad	108"	Bench	25	3-Cone	
100	UConn	Skyler Bell	WR	5'11.5"	192 lbs	40-Yard	4.40	Vertical	41.0"	Broad	133"	Shuttle		3-Cone	
101	Florida	Jake Slaughter	IOL	6'5"	303 lbs	40-Yard	5.10	Vertical	32.5"	Broad	110"	Bench		Arm	32.375"
102	Oklahoma	Gracen Halton	DL	6'2.5"	293 lbs	40-Yard	4.82	Vertical	36.5"	Broad	114"	Bench		3-Cone	8.09
103	Oregon	Isaiah World	OT	6'8"	312 lbs
104	Michigan	Jaishawn Barham	LB	6'3.5"	240 lbs	40-Yard	4.64	Vertical	33.0"	Broad	123"	Shuttle		3-Cone	
105	Florida	Austin Barber	OT	6'7"	318 lbs	40-Yard	5.12	Vertical	32.0"	Broad	111"	Bench		Arm	33.125"
106	Auburn	Keyron Crawford	EDGE	6'4.5"	253 lbs	40-Yard		Vertical		Broad		Bench		3-Cone	
107	TCU	Bud Clark	S	6'1"	188 lbs	40-Yard	4.41	Vertical	38.0"	Broad	127"	Shuttle		3-Cone	
108	Miami	Markel Bell	OT	6'9.5"	346 lbs	40-Yard	5.36	Vertical		Broad		Bench		Arm	36.375"
109	Cincinnati	Dontay Corleone	DL	6'0.5"	340 lbs	40-Yard		Vertical		Broad		Bench		3-Cone	
110	Miami	Carson Beck	QB	6'5"	233 lbs	40-Yard		Vertical		Broad		Shuttle		Hand	10.0"
111	Mississippi State	Brenen Thompson	WR	5'9.5"	164 lbs	40-Yard	4.26	Vertical		Broad		Shuttle		3-Cone	
112	Penn State	Nicholas Singleton	RB	6'0.5"	219 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
113	Texas	Jack Endries	TE	6'4.5"	245 lbs	40-Yard	4.62	Vertical	36.0"	Broad	119"	Shuttle		3-Cone	
114	Notre Dame	Billy Schrauth	IOL	6'5"	310 lbs	40-Yard		Vertical		Broad		Bench		Arm	32.875"
115	Texas A&M	Taurean York	LB	5'11"	226 lbs	40-Yard		Vertical		Broad		Shuttle	4.48	3-Cone	7.32
116	LSU	Harold Perkins Jr.	LB	6'1"	223 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
117	NC State	Justin Joly	TE	6'3.5"	241 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
118	Penn State	Zane Durant	DL	6'1"	290 lbs	40-Yard	4.75	Vertical	33.5"	Broad	112"	Bench		3-Cone	
119	Indiana	Mikail Kamara	EDGE	6'1"	265 lbs
120	Alabama	Parker Brailsford	IOL	6'2"	289 lbs	40-Yard	4.95	Vertical	32.5"	Broad	118"	Bench		Arm	32.0"
121	Duke	Brian Parker II	OT	6'5.5"	309 lbs	40-Yard	5.14	Vertical		Broad	109"	Bench		Arm	32.875"
122	Penn State	Drew Shelton	OT	6'5"	313 lbs	40-Yard	5.16	Vertical	31.0"	Broad	112"	Bench		Arm	33.375"
123	South Carolina	Jalon Kilgore	S	6'1.5"	210 lbs	40-Yard	4.40	Vertical	37.0"	Broad	130"	Shuttle	4.32	3-Cone	
124	Boise State	Kage Casey	OT	6'5.5"	310 lbs	40-Yard	5.20	Vertical		Broad		Bench		Arm	32.75"
125	Georgia	Daylen Everette	CB	6'1.5"	196 lbs	40-Yard	4.38	Vertical	37.5"	Broad	124"	Shuttle		3-Cone	
126	Stanford	Sam Roush	TE	6'6"	267 lbs	40-Yard	4.70	Vertical	38.5"	Broad	126"	Shuttle	4.37	3-Cone	7.08
127	Miami	CJ Daniels	WR	6'2.5"	202 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
128	Texas A&M	Nate Boerkircher	TE	6'5.5"	245 lbs	40-Yard		Vertical		Broad		Shuttle	4.40	3-Cone	
129	Ole Miss	De'Zhaun Stribling	WR	6'2"	207 lbs	40-Yard	4.36	Vertical	36.0"	Broad	127"	Shuttle		3-Cone	
130	Penn State	Kaytron Allen	RB	5'11.5"	216 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
131	Michigan	Rayshaun Benny	DL	6'3.5"	298 lbs	40-Yard		Vertical		Broad		Bench		3-Cone	
132	Alabama	Domani Jackson	CB	6'1"	194 lbs	40-Yard	4.41	Vertical		Broad		Shuttle		3-Cone	
133	Penn State	Drew Allar	QB	6'5.5"	228 lbs	40-Yard		Vertical		Broad		Shuttle		Hand	9.875"
134	Boston College	Jude Bowry	OT	6'5"	314 lbs	40-Yard	5.08	Vertical	34.5"	Broad	115"	Bench		Arm	33.75"
135	Baylor	Josh Cameron	WR	6'1.5"	220 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
136	Texas Tech	Skyler Gill-Howard	DL	6'0.5"	280 lbs	40-Yard		Vertical		Broad		Bench	27	3-Cone	
137	Wake Forest	Demond Claiborne	RB	5'10"	188 lbs	40-Yard	4.37	Vertical		Broad	122"	Shuttle		3-Cone	
138	Texas Tech	Reggie Virgil	WR	6'2.5"	187 lbs	40-Yard	4.57	Vertical	36.0"	Broad	127"	Shuttle		3-Cone	
139	Texas A&M	Dametrious Crownover	OT	6'7.5"	319 lbs	40-Yard	5.14	Vertical		Broad		Bench		Arm	35.375"
140	Iowa	Logan Jones	IOL	6'3"	299 lbs	40-Yard	4.90	Vertical	32.0"	Broad	110"	Bench		Arm	30.75"
141	Utah	Lander Barton	LB	6'4.5"	233 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
142	Indiana	Louis Moore	S	5'11"	191 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
143	Oregon	Alex Harkey	OT	6'5.5"	308 lbs	40-Yard		Vertical		Broad		Bench		Arm	31.75"
144	Georgia	Oscar Delp	TE	6'5"	245 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
145	Florida	Tyreak Sapp	EDGE	6'2.5"	273 lbs	40-Yard		Vertical		Broad		Bench	28	3-Cone	
146	Alabama	Tim Keenan III	DL	6'1"	327 lbs	40-Yard	5.31	Vertical	30.0"	Broad	100"	Bench	21	3-Cone	
147	Kentucky	Jager Burton	IOL	6'4"	312 lbs	40-Yard	4.94	Vertical	28.0"	Broad	111"	Bench		Arm	32.5"
148	Arkansas	Fernando Carmona Jr.	OT	6'4.5"	316 lbs	40-Yard	5.22	Vertical	29.0"	Broad	103"	Bench		Arm	32.125"
149	Missouri	Kevin Coleman Jr.	WR	5'10.5"	179 lbs	40-Yard	4.49	Vertical	38.5"	Broad	126"	Shuttle		3-Cone	
150	Illinois	J.C. Davis	OT	6'4.5"	322 lbs	40-Yard	5.16	Vertical	30.5"	Broad	99"	Bench		Arm	34.25"
151	Texas A&M	Trey Zuhn III	OT	6'6.5"	312 lbs	40-Yard		Vertical		Broad		Bench		Arm	32.5"
152	Washington	Ephesians Prysock	CB	6'3.5"	196 lbs	40-Yard	4.45	Vertical	39.0"	Broad	124"	Shuttle		3-Cone	
153	Notre Dame	Eli Raridon	TE	6'6"	245 lbs	40-Yard	4.62	Vertical	36.0"	Broad	123"	Shuttle		3-Cone	
154	Washington	Tacario Davis	CB	6'4"	194 lbs	40-Yard	4.41	Vertical	37.0"	Broad	123"	Shuttle		3-Cone	
155	Cincinnati	Joe Royer	TE	6'5"	247 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
156	North Dakota State	Bryce Lance	WR	6'3.5"	204 lbs	40-Yard	4.34	Vertical	41.5"	Broad	133"	Shuttle	4.15	3-Cone	7.00
157	Iowa	Beau Stephens	IOL	6'5.5"	315 lbs	40-Yard	5.35	Vertical	28.0"	Broad	102"	Bench		Arm	31.125"
158	Clemson	DeMonte Capehart	DL	6'5"	313 lbs	40-Yard	4.85	Vertical	33.5"	Broad	107"	Bench		3-Cone	
159	California	Hezekiah Masses	CB	6'0.5"	179 lbs	40-Yard	4.46	Vertical	31.5"	Broad	119"	Shuttle		3-Cone	
160	Texas A&M	Ar'maj Reed-Adams	IOL	6'6"	314 lbs	40-Yard	5.28	Vertical	29.5"	Broad	110"	Bench		Arm	34.375"
161	Notre Dame	Aamil Wagner	OT	6'6"	306 lbs	40-Yard	5.01	Vertical	29.5"	Broad	108"	Bench		Arm	34.5"
162	Iowa	TJ Hall	CB	6'1"	189 lbs	40-Yard	4.59	Vertical	36.0"	Broad		Shuttle	4.19	3-Cone	7.19
163	Utah	Dallen Bentley	TE	6'4"	253 lbs	40-Yard	4.62	Vertical	35.0"	Broad	118"	Shuttle	4.42	3-Cone	
164	Texas	Michael Taaffe	S	6'0"	190 lbs	40-Yard	4.50	Vertical		Broad		Shuttle		3-Cone	
165	Iowa	Max Llewellyn	EDGE	6'5.5"	258 lbs	40-Yard	4.81	Vertical	32.5"	Broad	115"	Bench		3-Cone	
166	LSU	Aaron Anderson	WR	5'8"	191 lbs	40-Yard		Vertical	30.0"	Broad	113"	Shuttle		3-Cone	
167	TCU	Eric McAlister	WR	6'3.5"	194 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
168	Clemson	Cade Klubnik	QB	6'2.5"	207 lbs	40-Yard		Vertical		Broad		Shuttle		Hand	9.25"
169	Texas A&M	Albert Regis	DL	6'1.5"	295 lbs	40-Yard	4.88	Vertical	34.0"	Broad	116"	Bench		3-Cone	7.77
170	Texas	DJ Campbell	IOL	6'3"	313 lbs	40-Yard	5.01	Vertical	26.5"	Broad	104"	Bench		Arm	34.25"
171	Alabama	Jaeden Roberts	IOL	6'5.5"	333 lbs	40-Yard		Vertical		Broad		Bench		Arm	33.375"
172	Texas	Ethan Burke	EDGE	6'6"	259 lbs
173	Indiana	Aiden Fisher	LB	6'1"	232 lbs	40-Yard		Vertical		Broad		Shuttle		3-Cone	
174	Baylor	Sawyer Robertson	QB	6'4"	216 lbs	40-Yard	4.64	Vertical	37.5"	Broad	123"	Shuttle	4.46	Hand	9.375"
175	Ole Miss	Zxavian Harris	DL	6'8"	330 lbs	40-Yard		Vertical		Broad		Bench		3-Cone	
176	Georgia Tech	Eric Rivers	WR	5'10"	176 lbs	40-Yard	4.35	Vertical	37.0"	Broad	127"	Shuttle		3-Cone	
177	USC	Bishop Fitzgerald	S	5'11"	201 lbs	40-Yard	4.55	Vertical	33.0"	Broad		Shuttle		3-Cone	
178	Michigan	Marlin Klein	TE	6'6"	248 lbs	40-Yard	4.61	Vertical	36.0"	Broad	117"	Shuttle		3-Cone	7.42
179	Virginia	J'Mari Taylor	RB	5'10"	199 lbs	40-Yard		Vertical	34.5"	Broad	115"	Shuttle		3-Cone	"""


def parse_height_to_inches(height_str):
    """Convert 6'2.5" to 74.5"""
    if not height_str:
        return None
    match = re.match(r"(\d+)'(\d+\.?\d*)", height_str.replace('"', ''))
    if match:
        feet = int(match.group(1))
        inches = float(match.group(2))
        return feet * 12 + inches
    return None


def parse_weight(weight_str):
    """Convert '241 lbs' to 241"""
    if not weight_str:
        return None
    match = re.match(r"(\d+)", weight_str)
    if match:
        return int(match.group(1))
    return None


def clean_value(val):
    """Clean measurement values - remove quotes/inches markers."""
    if not val or val.strip() == '':
        return ''
    val = val.strip().replace('"', '').replace("'", '')
    try:
        return str(float(val))
    except ValueError:
        return ''


def main():
    print("PARSING 2026 COMBINE DATA")
    print("=" * 70)

    rows = []

    for line in RAW_DATA.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        parts = line.split('\t')
        if len(parts) < 5:
            continue

        rank = parts[0].strip()
        school = parts[1].strip()
        name = parts[2].strip()
        position = parts[3].strip()
        height = parts[4].strip()
        weight = parts[5].strip() if len(parts) > 5 else ''

        # Parse measurables from remaining tab-separated pairs
        forty = ''
        vertical = ''
        broad = ''
        shuttle = ''
        three_cone = ''
        bench = ''
        hand = ''
        arm = ''

        # Remaining parts come in label/value pairs
        i = 6
        while i < len(parts) - 1:
            label = parts[i].strip().lower()
            value = parts[i + 1].strip() if i + 1 < len(parts) else ''

            if '40' in label:
                forty = clean_value(value)
            elif 'vertical' in label or 'vert' in label:
                vertical = clean_value(value)
            elif 'broad' in label:
                broad = clean_value(value)
            elif 'shuttle' in label:
                shuttle = clean_value(value)
            elif '3-cone' in label or 'cone' in label:
                three_cone = clean_value(value)
            elif 'bench' in label:
                bench = clean_value(value)
            elif 'hand' in label:
                hand = clean_value(value)
            elif 'arm' in label:
                arm = clean_value(value)

            i += 2

        rows.append({
            'Year': 2026,
            'Name': name,
            'College': school,
            'POS': position,
            'Big_Board_Rank': rank,
            'Height': height,
            'Height_in': parse_height_to_inches(height) or '',
            'Weight_lbs': parse_weight(weight) or '',
            '40_Yard': forty,
            'Bench_Press': bench,
            'Vert_Leap_in': vertical,
            'Broad_Jump_in': broad,
            'Shuttle': shuttle,
            '3Cone': three_cone,
            'Hand_Size_in': hand,
            'Arm_Length_in': arm,
        })

    # Save ALL positions
    output_all = os.path.join(OUTPUT_DIR, "combine_2026_all.csv")
    fieldnames = list(rows[0].keys())
    with open(output_all, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Save fantasy-only
    fantasy_pos = {'QB', 'RB', 'WR', 'TE'}
    fantasy_rows = [r for r in rows if r['POS'] in fantasy_pos]

    output_fantasy = os.path.join(OUTPUT_DIR, "combine_2026_fantasy.csv")
    with open(output_fantasy, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fantasy_rows)

    print(f"  Total prospects parsed: {len(rows)}")
    print(f"  Fantasy-relevant (QB/RB/WR/TE): {len(fantasy_rows)}")
    print(f"\n  Saved:")
    print(f"    {output_all}")
    print(f"    {output_fantasy}")

    # Quick summary
    print(f"\n  Fantasy breakdown:")
    for pos in ['QB', 'RB', 'WR', 'TE']:
        count = sum(1 for r in fantasy_rows if r['POS'] == pos)
        with_40 = sum(1 for r in fantasy_rows if r['POS'] == pos and r['40_Yard'])
        print(f"    {pos}: {count} total, {with_40} with 40-yard")

    print(f"\n{'=' * 70}")
    print("DONE")


if __name__ == "__main__":
    main()