0	book
1	drink
2	computer
3	before
4	chair
5	go
6	clothes
7	who
8	candy
9	cousin
10	deaf
11	fine
12	help
13	no
14	thin
15	walk
16	year
17	yes
18	all
19	black
20	cool
21	finish
22	hot
23	like
24	many
25	mother
26	now
27	orange
28	table
29	thanksgiving
30	what
31	woman
32	bed
33	blue
34	bowling
35	can
36	dog
37	family
38	fish
39	graduate
40	hat
41	hearing
42	kiss
43	language
44	later
45	man
46	shirt
47	study
48	tall
49	white
50	wrong
51	accident
52	apple
53	bird
54	change
55	color
56	corn
57	cow
58	dance
59	dark
60	doctor
61	eat
62	enjoy
63	forget
64	give
65	last
66	meet
67	pink
68	pizza
69	play
70	school
71	secretary
72	short
73	time
74	want
75	work
76	africa
77	basketball
78	birthday
79	brown
80	but
81	cheat
82	city
83	cook
84	decide
85	full
86	how
87	jacket
88	letter
89	medicine
90	need
91	paint
92	paper
93	pull
94	purple
95	right
96	same
97	son
98	tell
99	thursday

## 2026-05-26 Dev A verification

Checkpoint: `checkpoints/best.pt`

Config:
- raw live window: `(100, 144)`
- model input after presence bits + velocity: `(100, 292)`
- classes: 100

Evaluation:
- validation, 4-crop TTA: Top-1 169/238 = 71.01%, Top-3 84.87%, Top-5 86.55%
- test, 4-crop TTA: Top-1 127/201 = 63.18%, Top-3 81.59%, Top-5 86.07%

Export:
- ONNX: `exports/asl_model.onnx`
- label map: `exports/label_map.json`
- metadata: `exports/export_meta.json`
- backend local copy: `../backend/models/asl_model.onnx` and `../backend/models/label_map.json`
