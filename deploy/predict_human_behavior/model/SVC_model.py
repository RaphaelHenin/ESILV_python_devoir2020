from sklearn import svm
import pandas as pd
import joblib

# Chargement du nom des colonnes et suppression des espaces
index_column = pd.read_csv('../../../features.txt', header=None, sep="\n")
index_column = index_column.loc[:, 0].str.strip()
# Chargement des données de train
trainX_df = pd.read_csv('../../../Train/X_train.txt', sep=" ", header=None)
trainY_df = pd.read_csv('../../../Train/y_train.txt',
                        names={"id_label"}, header=None)

# Chargement des données de test
testX_df = pd.read_csv('../../../Test/X_test.txt', sep=" ", header=None)
testY_df = pd.read_csv('../../../Test/y_test.txt',
                       names={"id_label"}, header=None)

# Insertion des noms des colonnes
trainX_df.columns = index_column.values
testX_df.columns = index_column.values

# Chargement des labels des activités
activity_label_df = pd.read_csv(
    '../../../activity_labels.txt', sep="\n", names={"activity"}, header=None)
activity_label_df['id_label'] = activity_label_df['activity'].str[0:2].astype(
    'int64')
activity_label_df['activity'] = activity_label_df['activity'].str[2:]
activity_label_df['activity'] = activity_label_df['activity'].str.strip(
).astype('category')

# Merge
trainY_df = pd.merge(trainY_df, activity_label_df,
                     on='id_label', how='left', sort=False)
testY_df = pd.merge(testY_df, activity_label_df,
                    on='id_label', how='left', sort=False)

# Create your models here.
model = svm.SVC(C=16, kernel='linear')
model = model.fit(trainX_df, trainY_df['id_label'])

joblib.dump(model, "./SVC.pkl", compress=True)

model_ld = joblib.load("./SVC.pkl")

id_predict = model_ld.predict([[0.03091390114285231, -0.008926886096823239, 0.04038222890711229, -0.9385036401585529, -0.9446259898253684, -0.7593342935050583, -0.9523983334771264,
                        -0.9502807547473435, -0.802483410403077, -0.7570991203421881, -0.7333977187438867, -
                        0.4079602513726583, 0.7285110199854334, 0.6582664805843173, 0.6291692577948003, -
                        0.9133057590511835,
                        -0.9979660020375915, -0.9986833493660462, -0.9758179391357794, -0.9754396044554765, -
                        0.9667931503079801, -0.9143962266301497, -
                        0.3638215871989212, -0.4208966992183205, 0.1817720865689756,
                        0.517927592013437, -0.4036355193796901, 0.4493250534646054, -0.1475446797665396, 0.2231503449072767, -
                        0.1901429689138278, 0.1558993110055844, 0.05403484158870775, 0.0531561643290619,
                        -3.746562579909352e-05, -0.07788849920985308, 0.02858544288773346, 0.07765907729415811, -
                        0.4906156666901278, -0.7119640871429516, 0.9331510397395832, -
                        0.2773603493364716, 0.1154688298016364,
                        -0.9400117857650483, -0.9404602351672032, -0.7141479013569905, -0.9456994201489609, -
                        0.9422842872895677, -0.7447003545469033, 0.9060825875842768, -
                        0.2793171307296632, 0.1528951867195498,
                        0.9444614024290725, -0.2602485894796517, -0.07648386743883717, -0.01631454406021937, 0.8219321708857537, -
                        0.8642964556462403, -0.9677956775799085, -
                        0.9651743515166069, -0.9538413285779899,
                        -0.850600809846409, -0.4719446782705673, -1, 0.1321396475353758, -0.4346419144856691, 0.4366703231638911, -
                        0.4391054858831203, 0.4419497747396852, -
                        0.7688570207804795, 0.7725977501171459,
                        -0.776722256796994, 0.7805093509218983, -0.9815505971527755, 0.9842598934836881, -
                        0.9867120746983097, 0.9878265264077741, 0.9806393885474847, -
                        0.9962174733209269, -0.9601406389500177,
                        0.07204600702976638, 0.0457544011017863, -0.1314457891214891, -0.906682758014516, -
                        0.9380163897710612, -0.935935827672327, -0.9160326567412938, -
                        0.9367254603855806, -0.9490537889711499,
                        -0.9032241537690432, -0.9498183292476219, -0.8914034739092879, 0.8984793545667791, 0.9501816438783206, 0.946152786975107, -
                        0.9308047417691804, -0.9950459267294768, -0.9974955077029623,
                        -0.9970156048278576, -0.9364575059822105, -0.9468741343916546, -0.9682441526293516, -
                        0.08517415145592266, -0.3102630367003146, -
                        0.4897322161891108, 0.5080165454687009, -0.3397138947231827,
                        0.5847900559955122, 0.4180322773692125, 0.2862571300694827, -
                        0.04503927861088541, 0.2750570282424627, 0.2910722815087194, 0.03519514149476222, 0.079089181499576, 0.009473936871008171,
                        0.1207895904124499, -0.2733794154312873, -0.01463350411690167, -0.3733534108902208, 0.1199761559539552, -
                        0.03293900834792596, -0.01083446967064972, -
                        0.8830908515732687, -0.8161490126612015,
                        -0.9500802718977778, -0.8886150599844537, -0.8577764414185091, -0.9535111187232572, -
                        0.7559680345348243, -0.7159972898628676, -
                        0.6687691744127293, 0.8440509692033762, 0.8063228648178504,
                        0.8109771408153192, -0.7964427141043902, -0.9796162641133277, -0.9829000656343006, -
                        0.9954258889328465, -0.8931434008242012, -
                        0.9060684072465594, -0.9581343508755605, 0.7740327904708151,
                        -0.2846095942278191, 0.3654526181414828, 0.1814194090447894, -0.2440978508359321, 0.2432597089768886, -
                        0.1834848161485012, 0.04434599437976017, -
                        0.09684399630869345, 0.04124078156418332,
                        0.3106459080987627, 0.2620014081948097, -0.3154083635868323, 0.2968060551447289, -
                        0.0651126625030849, 0.6510354552637556, -
                        0.2359531310611729, -0.300199976092408, -0.2048962143984827,
                        -0.1744877061618777, -0.09338933967644458, -0.9010405206254601, -0.9108600523378149, -
                        0.9395396275102473, -0.9101677698330515, -
                        0.9273641171596608, -0.953765806054418, -0.8904452704629463,
                        -0.9133263426833513, -0.8974370081717655, 0.9047422741591931, 0.9173134178387135, 0.9476121962263386, -
                        0.929675953440413, -0.9946854776487251, -
                        0.9957905663330307, -0.9978143987961576,
                        -0.9364578684897862, -0.9588787118705886, -0.9706854980539517, 0.03661912040069937, 0.06476706639690999, -
                        0.1971260538830064, 0.210460430848826, 0.02662680840530229, 0.1110125561727373,
                        0.2881094005934457, 0.1384865467486021, -0.01861244272394547, 0.05798281847368991, 0.2254505872495176, 0.3642039616084283, -
                        0.3489269252086173, 0.3252287512989445, 0.2443108207909794,
                        0.1041313725491369, -0.1639455677455807, -0.2926367291125744, -0.8999522400005415, -
                        0.7739326786561038, -0.8043340626449487, -
                        0.761637872702188, -0.9856389971831475, -0.8999522400005415,
                        -0.9888621143474495, -0.9024762754473019, 0.2629378324327596, 0.1019446371395418, -
                        0.2431181429704169, 0.6447236591408063, -0.560866683551417, -
                        0.8999522400005415, -0.7739326786561038,
                        -0.8043340626449487, -0.761637872702188, -0.9856389971831475, -0.8999522400005415, -
                        0.9888621143474495, -
                        0.9024762754473019, 0.2629378324327596, 0.1019446371395418, -0.2431181429704169,
                        0.6447236591408063, -0.560866683551417, -0.9305240281056446, -0.8959684206196013, -
                        0.900342052332748, -0.902671655644413, -0.9750110913352466, -
                        0.9305240281056446, -0.9956168353601917,
                        -0.9141573080985916, -0.1294728864092357, 0.2603468084385137, -0.3798252101784621, 0.1975558386087575, -
                        0.1306837443160881, -0.7954937853898916, -
                        0.7620732187965074, -0.7948354261624644,
                        -0.719262243882184, -0.860659311188384, -0.7954937853898916, -0.9741519264186922, -
                        0.8563582671003438, 0.7136070365178702, 0.07108362922799127, -
                        0.2905899917630047, 0.2491940639911863,
                        0.008354533702243705, -0.9252451607610636, -0.8943436102757514, -0.9001466845812274, -
                        0.9167370801788042, -0.9763666496093859, -
                        0.9252451607610636, -0.9958244128158091, -0.9118374965315184,
                        0.3316543102571707, 0.5410613802370505, -0.4984127474356624, -0.01985754328597422, 0.08786251104193643, -
                        0.9185353069460669, -0.9182131912905719, -
                        0.7892636892263446, -0.9481441429073049,
                        -0.9542015757464845, -0.7690659994731305, -0.9306181392000218, -0.9252535064408023, -
                        0.726087060846026, -0.9683326409927141, -0.9647512112927171, -
                        0.7988848342049312, -0.9661280737094377,
                        -0.9844523057271841, -0.952100938218399, -0.8650631768884978, -0.9978441606999687, -
                        0.9980854769226886, -0.9684884135402175, -
                        0.9056435949352742, -0.9338124548144021, -0.8720310338755473,
                        -0.3396732713715448, -0.4858032356185907, -0.1662576548849375, -1, -1, -1, 0.1282224303876582, 0.149118541205882, -
                        0.4033839445652052, -0.5721046109351515, -
                        0.8953226231617459, -0.3822352008631715,
                        -0.6790095738445087, 0.1976889648676732, -0.1247681780909428, -0.9985060592341875, -
                        0.9979434734202679, -0.9955978778125502, -
                        0.9950355978299396, -0.9959768993300702, -0.9914885308063356,
                        -0.992131099068687, -0.9997751177920647, -0.998203515980335, -0.99474673387441, -0.9943079084715906, -
                        0.9946931907479997, -0.9980206799812816, -
                        0.9937258810825598, -0.9987296589242712,
                        -0.9979761522063216, -0.9989229009855432, -0.9964001866057919, -0.9975679714153203, -
                        0.9950572067242812, -0.9973517415997432, -
                        0.9982380474926725, -0.9982920820690527, -0.997923448425224,
                        -0.9963602714686516, -0.9976429735328916, -0.9982163046863276, -0.9962253888740256, -
                        0.9697372674197318, -0.994092038944523, -0.9953234737520752, -
                        0.9989694449522466, -0.9979050977195052,
                        -0.9949296233966826, -0.9892604868181947, -0.9815934988590913, -0.9689584945469225, -
                        0.9966743953579018, -0.9970186380216172, -
                        0.9864232111563952, -0.9685504440506304, -0.9984312662434426,
                        -0.8996331613640829, -0.937485000244586, -0.9235007423269892, -0.9244291311084237, -
                        0.9432103754474016, -0.9473166689660065, -
                        0.89661454912731, -0.9383091070710939, -0.9427035535230063,
                        -0.9486374453908667, -0.9583254240084939, -0.9587333853619826, -0.9438865183947495, -
                        0.9873026273954517, -0.9784767280135628, -
                        0.9057235698228904, -0.9950360850308671, -0.9974993089829107,
                        -0.9970306999073901, -0.8862611112833655, -0.9358644776849802, -0.9529317311561105, -0.4706616034530879, -
                        0.672171802219095, -0.5962740431085914, -
                        0.52, 0.08000000000000007, 0.3200000000000001,
                        0.3194396018654739, 0.1431039312678153, -0.01882813496157865, -0.5932957307667939, -
                        0.9247359915911411, -0.6884598971603044, -
                        0.9427571578006404, -0.6315878322030913, -0.9146406234448515,
                        -0.9990115631765639, -0.9976308549714603, -0.995984772979356, -0.9947083149541961, -
                        0.9955715771467559, -0.9878764345777176, -
                        0.9864460144531079, -0.9982051662530462, -0.9980406888593472,
                        -0.994497392468024, -0.9920554297913922, -0.9863355389430242, -0.9968919169394902, -
                        0.9907639165262392, -0.999409156453233, -0.9986480646011373, -
                        0.9987487522967076, -0.9964035739872634,
                        -0.9971775930046012, -0.9945286300298363, -0.9979197944473005, -0.9999697005523838, -
                        0.9986417095358132, -0.997399635185765, -
                        0.9954063029096392, -0.998182401237619, -0.9984630645941902,
                        -0.9959958081826253, -0.9936378575837243, -0.997835615659062, -0.9971293071424188, -
                        0.9987438414777747, -0.9966949399942318, -
                        0.9951096613894537, -0.9984811268616411, -0.9990934829092509,
                        -0.9959223643644333, -0.9979512595016876, -0.9960171931778703, -0.9984556074673918, -
                        0.9962672327429929, -0.9976948389929294, -
                        0.8235719135128713, -0.8079246298303293, -0.9177593590875502,
                        -0.9032579763113933, -0.8226963195684236, -0.9637185736803142, -0.8739348548622755, -
                        0.831833887407869, -0.9410139566617426, -0.9118707764133491, -
                        0.8792822861159526, -0.9705629888502667,
                        -0.8785995844283935, -0.9483311076617874, -0.9196764656817156, -0.8284564616280817, -
                        0.9929495304324586, -0.9826630723715216, -
                        0.9985741360722351, -0.8787744711773703, -0.8381441465747861,
                        -0.9291399493897948, 0.000758148056667407, 0.2001436797950276, -0.2533841568600872, -1, -
                        0.935483870967742, -0.935483870967742, 0.2005046729556648, -
                        0.04390420274227302, 0.3589099501108479,
                        -0.3966218684235262, -0.6967687089274699, -0.3950200534522101, -0.7853731027209898, -
                        0.4978224699471603, -0.7831997007362812, -
                        0.994629084650093, -0.9904695656125638, -0.992777067142068,
                        -0.9956799893582304, -0.9870793445596416, -0.9866508894686165, -0.9841148806631836, -
                        0.9852502143389775, -0.9935527351714183, -
                        0.9924289859004025, -0.9856066858718655, -0.9846173516970385,
                        -0.9932692148294086, -0.992726366750557, -0.9825653681733066, -0.9948521968641821, -
                        0.9978192089154001, -0.9948430627638796, -
                        0.992409951294188, -0.98812337578959, -0.9904333476221433,
                        -0.9879612706720264, -0.9817908807000156, -0.9963991883831106, -0.9914527048045365, -
                        0.9880604774942483, -0.9820483905738913, -
                        0.9933800130097545, -0.9992414715579907, -0.9983659235263092,
                        -0.9984672442406484, -0.9983752279014682, -0.9989058647660959, -0.9958936114064334, -
                        0.9931283639412717, -0.9954723029588953, -
                        0.9988588787366445, -0.9977576459829683, -0.9980961617614558,
                        -0.9941476464546596, -0.998714750940944, -0.998290193192808, -0.7903915811098655, -
                        0.7812038684346941, -0.7264002145496404, -
                        0.8399827678420891, -0.9418233487573212, -0.7903915811098655,
                        -0.9730080393622529, -0.8723399849596911, -0.1745928792793949, -1, -0.4483788116965972, 0.005536588584989843, -
                        0.3835552315357723, -0.8949515120312651, -
                        0.8968076718939383, -0.8890153974077192,
                        -0.9288655305395519, -0.8980964352136654, -0.8949515120312651, -0.9934711022975858, -0.9218358696294603, -
                        0.4846192900389216, -1, -0.03175381543350375, -
                        0.2551142300142815, -0.6986712066117873,
                        -0.770610000090742, -0.7971049689887836, -0.7692225684201997, -0.8342661172378463, -
                        0.9403593366447155, -0.770610000090742, -
                        0.9709580247757197, -0.798402639387964, 0.1794352273668083,
                        -1, -0.05457564479879962, -0.4968200808929376, -0.7647550719517966, -0.8901688903646774, -
                        0.907479737199415, -0.8955180631910915, -0.9179533684180817, -
                        0.9098251813639962, -0.8901688903646774,
                        -0.9941054297032158, -0.8980968545120932, -0.2348152901855691, -1, 0.1228301368024483, -
                        0.3456843742823689, -
                        0.7090872022899664, 0.006462402864424899, 0.162919820165899, -0.8258856226927339,
                        0.2711514521489811, -0.7205591038773122, 0.2767794147286349, -0.0510740294455464]])
print(activity_label_df['activity'].loc[id_predict-1].astype(str))
