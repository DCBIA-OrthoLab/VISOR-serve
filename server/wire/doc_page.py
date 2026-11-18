"""Per-tool documentation, served at `GET /doc/{tool}`.

Why here and not a README: a README explains a tool to whoever is about to
change it, from inside its own repository. This explains a tool to whoever is
about to RUN it against a server -- what it does to a note, what it costs, and
where it can refuse -- and it is the deployment that knows the second of those,
because the numbers come from this machine's own measurements.

One `DOCS` entry per tool, so adding one is adding a dict key. A tool with no
entry answers 404 rather than an empty page: a blank document reads as "there
is nothing to say", and what is true is "nobody has written it".

Self-contained, like the status and benchmark pages, and for the same reason:
this server runs on networks with no internet, so a stylesheet or a font from
a CDN is a broken page rather than a degraded one.

**The prose is French; everything around it is not.** `CLAUDE.md` puts every
file in this repository in English, and that rule is about the code -- the
identifiers, the comments, the messages a developer reads while changing this.
A document is the one thing here written FOR its reader rather than for
whoever maintains it, and this one's reader asked for French. So the strings
below are French and the module around them stays English, which keeps what
the rule protects.
"""

_STYLE = """
:root {
  --bg:#f6f6f4; --panel:#fffffe; --edge:#e2e1dc; --ink:#1d1e1b;
  --soft:#6b6d66; --faint:#93958d; --sunk:#efeeea;
  --accent:#2f6b55; --warn:#c9821f; --alert:#a2543a;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",sans-serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#17181a; --panel:#1f2023; --edge:#32343a; --ink:#e8e8e4;
  --soft:#a0a29b; --faint:#74766f; --sunk:#26282c;
  --accent:#4e9c7c; --warn:#d89a3f; --alert:#d08a6c; }}
:root[data-theme="dark"]{
  --bg:#17181a; --panel:#1f2023; --edge:#32343a; --ink:#e8e8e4;
  --soft:#a0a29b; --faint:#74766f; --sunk:#26282c;
  --accent:#4e9c7c; --warn:#d89a3f; --alert:#d08a6c; }
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:15px/1.65 var(--sans);margin:0;
     padding:26px clamp(16px,4vw,48px) 96px}
.wrap{max-width:82ch;margin:0 auto}
header{border-bottom:1px solid var(--edge);padding-bottom:14px;margin-bottom:26px}
h1{font-size:22px;margin:0 0 4px;font-weight:600;letter-spacing:-.01em}
.sub{color:var(--soft);font-size:14px}
.crumb{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);
       margin-bottom:6px}
h2{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--soft);
   margin:36px 0 10px;font-weight:600}
h3{font-size:15px;margin:22px 0 6px;font-weight:600}
p{margin:0 0 12px}
code{font-family:var(--mono);font-size:.88em;background:var(--sunk);
     padding:1px 5px;border-radius:4px}
pre{background:var(--panel);border:1px solid var(--edge);border-radius:7px;
    padding:13px 15px;overflow-x:auto;margin:0 0 14px}
pre code{background:none;padding:0;font-size:12.5px;line-height:1.6}
table{border-collapse:collapse;width:100%;font-size:14px;margin:0 0 14px}
th{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--faint);
   text-align:left;font-weight:600;padding:0 12px 5px 0;white-space:nowrap}
td{padding:5px 12px 5px 0;border-top:1px solid var(--edge);
   font-variant-numeric:tabular-nums;vertical-align:top}
td.t{font-family:var(--mono);font-size:12.5px}
.num{text-align:right}
/* Where a phase actually runs. The bar's WIDTH is its share of the run, so a
   long host-bound phase is visibly long; the fill is the fraction of that
   phase with a kernel resident on the card. Two facts in one shape, because
   the question "what is on the GPU" is really two questions -- how much of
   the run, and how busy the card was while it happened. */
.where{display:flex;align-items:center;gap:9px}
.wbar{height:12px;border-radius:3px;background:var(--sunk);
      border:1px solid var(--edge);overflow:hidden;flex:0 0 auto}
.wbar i{display:block;height:100%;background:var(--accent)}
.wlab{font-size:11px;color:var(--faint);white-space:nowrap;
      font-variant-numeric:tabular-nums}
.wkey{display:flex;gap:16px;margin:0 0 12px;font-size:11.5px;color:var(--soft)}
.wkey span{display:flex;align-items:center;gap:6px}
.wsw{width:22px;height:10px;border-radius:2px;border:1px solid var(--edge)}
.wsw.gpu{background:var(--accent)} .wsw.host{background:var(--sunk)}

.note{border-left:2px solid var(--accent);padding:2px 0 2px 13px;margin:0 0 14px;
      color:var(--soft);font-size:14px}
.note b{color:var(--ink)}
.warnbox{border-left-color:var(--warn)}
.alertbox{border-left-color:var(--alert)}
.step{display:grid;grid-template-columns:26px 1fr;gap:12px;margin-bottom:16px}
.n{font-family:var(--mono);font-size:12px;color:var(--faint);
   border:1px solid var(--edge);border-radius:5px;height:24px;
   display:flex;align-items:center;justify-content:center}
.step p{margin:0 0 6px}
.step pre{margin-top:8px}
a{color:var(--accent)}
footer{margin-top:44px;padding-top:14px;border-top:1px solid var(--edge);
       color:var(--faint);font-size:12.5px}
"""

_CNE = """
<div class="crumb">Doc</div>
<h1>CNE &mdash; extraction de notes cliniques</h1>
<div class="sub">Des informations structurées à partir de texte libre, avec un
modèle de langage qui ne quitte jamais cette machine.</div>
</header>

<p>CNE est le seul tool du catalogue dont les poids sont un modèle de langage.
Il lit une note clinique, demande au modèle ce qu'elle contient, et réécrit le
résultat en JSON et en texte. <b>Rien ne sort de la machine</b> : le modèle est
un fichier sur disque et l'inférence est locale.</p>

<h2>Le déroulé d'un run</h2>

<div class="step"><div class="n">1</div><div>
<p><b>Découvrir les notes.</b> Un fichier ou un dossier, et trois extensions
acceptées &mdash; qui sont toutes les trois réellement lues, ce qui n'a pas
toujours été vrai.</p>
<pre><code>NOTE_EXTENSIONS = (".txt", ".pdf", ".docx")</code></pre>
<p>Un <code>.txt</code> est <i>décodé</i> et non supposé UTF-8 ; un
<code>.docx</code> est lu en entier et non paragraphe par paragraphe, ce qui
laissait tomber tout ce qui se trouve dans un tableau ; et le fichier verrou
<code>~$nom.docx</code> que Word laisse à côté d'un document ouvert est ignoré
&mdash; il porte l'extension, n'est pas un document, et fait lever le lecteur.</p>
</div></div>

<div class="step"><div class="n">2</div><div>
<p><b>Résoudre le modèle, et vérifier qu'il correspond au type de note.</b> Le
modèle est un unique fichier <code>.gguf</code> quantifié, choisi par son nom
dans le magasin du serveur &mdash; jamais téléversé avec la requête.</p>
<pre><code>MODEL_EXTENSION = ".gguf"
model_file = resolve_model_file(model)
hint = model_type_hint(model_file)      # "TMJ", lu dans le nom du fichier
if hint and hint != notes_type:         # un avertissement, pas un refus
    ...</code></pre>
<p>Un modèle TMJ à qui on donne des notes Ortho produit un <b>avertissement</b>
et non une erreur : le décalage regarde le déploiement, et le run peut très
bien être celui qu'on voulait.</p>
</div></div>

<div class="step"><div class="n">3</div><div>
<p><b>Construire la conversation.</b> Le type de note décide de deux choses
&mdash; la fenêtre de contexte et l'existence d'un prompt système &mdash; et
les deux sortent d'une <b>seule table, lue une fois</b>.</p>
<pre><code>CONTEXT_TOKENS = {"TMJ": 6144, "Ortho": 2048}
SYSTEM_PROMPTS  = {"TMJ": INSTRUCTION_TMJ, "Ortho": None}</code></pre>
<p>Ortho n'a <b>aucun</b> prompt système, et c'est volontaire : son modèle est
affiné pour répondre à partir de la note seule, et une instruction qu'il n'a
jamais vue à l'entraînement change ce qu'il émet.</p>
</div></div>

<div class="step"><div class="n">4</div><div>
<p><b>Interroger, une fois par note.</b> Un seul
<code>create_chat_completion</code> contre un 7B quantifié en 4 bits, sur CPU.</p>
<pre><code>temperature = 0.0        # la même note extrait la même réponse
max_tokens  = 2048
seed        = 0</code></pre>
<p>La température à zéro n'est pas un réglage de qualité. <b>Une extraction
clinique qu'on ne peut pas reproduire, on ne peut pas la vérifier.</b></p>
</div></div>

<div class="step"><div class="n">5</div><div>
<p><b>Lire la réponse, dans l'une de deux formes.</b> Le JSON est tenté
d'abord, découpé entre la première <code>{</code> et la dernière
<code>}</code> parce que les fine-tunes l'enveloppent dans une phrase. À
défaut, la réponse est lue en lignes <code>clé: valeur</code> &mdash; ce
qu'émet réellement le fine-tune TMJ publié : <b>46 champs, aucun JSON</b>.</p>
<pre><code>parse_extraction(answer)  ->  dict | None</code></pre>
<p><code>None</code> veut dire « aucune extraction ici », et l'appelant
n'écrit alors rien pour cette note plutôt que d'écrire quelque chose
d'inexploitable.</p>
</div></div>

<div class="step"><div class="n">6</div><div>
<p><b>Écrire.</b> Par note, un <code>Extraction_&lt;note&gt;.json</code> et un
<code>Extraction_&lt;note&gt;.txt</code>, plus un
<code>CNE_report.json</code> pour le lot.</p>
<p>Le nom de sortie porte le nom <i>complet</i> de la note, extension comprise :
avec trois extensions acceptées, <code>B_001.txt</code>,
<code>B_001.pdf</code> et <code>B_001.docx</code> dans un même dossier
écrivaient auparavant trois fois le même fichier.</p>
</div></div>

<h2>Où passe le temps</h2>

<p>Mesuré sur ce déploiement : <b>392 tokens de réponse en 38,9&ndash;41,5 s</b>,
soit 9,4 à 10,1 tokens par seconde. Ce n'est pas un problème de Python &mdash;
la boucle de tokens fait <b>moins de 0,2 %</b> du run.</p>

<table>
<tr><th>threads</th><th class="num">tokens/s</th><th></th></tr>
<tr><td class="t">14</td><td class="num">8,90</td><td></td></tr>
<tr><td class="t">28</td><td class="num">10,08</td><td>le défaut de la bibliothèque</td></tr>
<tr><td class="t">56</td><td class="num">6,42</td><td>tous les cœurs logiques, et plus lent</td></tr>
</table>

<div class="note"><b>Il est limité par la bande passante mémoire, pas par le
calcul.</b> Chaque token généré fait transiter les 4,4&nbsp;Go de poids depuis
la DRAM. Cette machine mesure 61,1&nbsp;Go/s ; le décodage en déplace
44,4&nbsp;Go/s, soit <b>73 % du plafond</b>. Aucun thread supplémentaire ne peut
déplacer ce chiffre, et au-delà de 28 il coûte.</div>

<div class="note alertbox"><b>Connu, ouvert :</b> le serveur accorde
actuellement à CNE sa part déclarée de threads plutôt que les 28 qu'il veut, ce
qui a mesuré <b>47,4&nbsp;s &rarr; 67,1&nbsp;s</b> sur la note de couverture. La
règle du grant de threads a été tirée d'un balayage de cinq autres tools, et
CNE &mdash; le seul qui scale vraiment &mdash; n'en faisait pas partie.</div>

<h2>Pourquoi il tourne sur CPU</h2>

<p>La build livrée de <code>llama-cpp-python</code> n'a aucun backend CUDA, et
c'est délibéré : le README rejette CUDA pour des raisons de reproductibilité.
Demander <code>device="cuda"</code> <b>lève</b> donc, plutôt que de retomber
silencieusement sur le CPU &mdash; un run qui ignore en silence le matériel
qu'on lui a demandé est un run dont les temps ne veulent rien dire.</p>

<div class="note warnbox">Les deux roues <code>nvidia-*</code> restent des
dépendances dures, parce que le passage documenté vers l'index CUDA doit rester
possible. Elles sont donc présentes dans un déploiement CPU, et il ne faut pas
les charger : le préchargement ouvrait <b>869&nbsp;Mo en
<code>RTLD_GLOBAL</code></b> et 149&nbsp;Mo résidents, pour rendre résolvables
des symboles que personne n'irait chercher. Le préchargement vérifie désormais
si la build installée lie le runtime CUDA.</div>

<h2>Quatre choses que le portage a changées</h2>

<table>
<tr><th>en amont</th><th>ici</th></tr>
<tr><td>Le type de note comparé par <code>notesType == "TMJ"</code> à un endroit
et <code>.upper() == "TMJ"</code> à un autre</td>
<td>Un <code>Literal["TMJ", "Ortho"]</code> validé avant l'entrée dans
<code>run</code>. <code>"tmj"</code> recevait l'instruction TMJ avec la fenêtre
Ortho de 2048 tokens &mdash; une note longue silencieusement tronquée, à cause
d'une différence d'orthographe.</td></tr>
<tr><td>Génération plafonnée à 500 tokens ; le JSON qui débordait était écrit
brut</td>
<td>Une réponse tronquée est un échec. <code>finish_reason</code> dit
exactement quand c'est arrivé.</td></tr>
<tr><td><code>temperature = 0.1</code></td>
<td><code>0.0</code>. La même note extraite deux fois donnait deux réponses
différentes.</td></tr>
<tr><td><code>dup2</code> de stderr vers <code>/dev/null</code> sans
<code>try/finally</code></td>
<td>Un gestionnaire de contexte. Un chargement qui levait laissait tout le
processus écrire sa trace dans le vide, définitivement.</td></tr>
</table>

<h2>Arguments</h2>

<table>
<tr><th>nom</th><th>type</th><th>défaut</th><th></th></tr>
<tr><td class="t">notes</td><td class="t">chemin</td><td>&mdash;</td>
    <td>une note, ou un dossier de notes</td></tr>
<tr><td class="t">notes_type</td><td class="t">TMJ | Ortho</td><td>&mdash;</td>
    <td>choisit la fenêtre de contexte et le prompt système</td></tr>
<tr><td class="t">model</td><td class="t">nom</td><td>&mdash;</td>
    <td>un bundle <code>.gguf</code> détenu par ce serveur</td></tr>
<tr><td class="t">max_tokens</td><td class="t">int</td><td class="num">2048</td>
    <td>une réponse tronquée est un échec, pas un résultat</td></tr>
<tr><td class="t">temperature</td><td class="t">float</td><td class="num">0.0</td>
    <td>reproductibilité, pas réglage</td></tr>
<tr><td class="t">context_tokens</td><td class="t">int</td><td class="num">0</td>
    <td>0 = le défaut du type de note</td></tr>
<tr><td class="t">seed</td><td class="t">int</td><td class="num">0</td><td></td></tr>
<tr><td class="t">device</td><td class="t">cpu | cuda</td><td class="t">cpu</td>
    <td>cuda lève sur une build sans le backend</td></tr>
</table>

<footer>Mesuré sur ce déploiement. Les chiffres viennent de la passe de
couverture et du profil de tokens, pas d'une spécification.</footer>
"""

_AMASSS = """
<div class="crumb">Doc</div>
<h1>AMASSS &mdash; segmentation des structures du crâne</h1>
<div class="sub">Neuf réseaux entraînés séparément, un par structure, passés
sur le même CBCT.</div>
</header>

<p>AMASSS prend un CBCT orienté et rend un masque par structure demandée
&mdash; mandibule, maxillaire, base du crâne, vertèbre cervicale, voies
aériennes supérieures, peau, et les trois masques associés. <b>Chaque structure
est un modèle nnUNet v2 distinct</b> : en demander cinq, c'est charger cinq
réseaux et faire cinq inférences complètes sur le même scan. Le bundle pèse
2,1&nbsp;Go, soit environ 236&nbsp;Mio par modèle.</p>

<h2>Le déroulé d'un run</h2>

<div class="step"><div class="n">1</div><div>
<p><b>Découvrir les scans.</b> Un volume ou un dossier, parcouru
récursivement. Un fichier qui ressemble à une sortie AMASSS précédente est
écarté, ce qui permet de relancer un dossier sur place sans qu'un second
passage réingère les résultats du premier.</p>
<pre><code>.nii  .nii.gz  .nrrd  .nrrd.gz  .gipl  .gipl.gz</code></pre>
<p>Aucun scan trouvé, ou aucun scan lisible : le run <b>refuse</b> au lieu de
réussir sur zéro patient.</p>
</div></div>

<div class="step"><div class="n">2</div><div>
<p><b>Résoudre le bundle.</b> Un sous-dossier par code de structure, choisi par
son nom dans le magasin du serveur &mdash; jamais téléversé avec la
requête.</p>
<pre><code>&lt;bundle&gt;/&lt;CODE&gt;/**/*__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth</code></pre>
<p>Une structure dont le bundle n'a pas les poids est <b>signalée dans le
rapport</b>, pas fatale : les autres sont segmentées. C'est l'absence de
<i>toute</i> structure utilisable qui fait refuser le run.</p>
</div></div>

<div class="step"><div class="n">3</div><div>
<p><b>Convertir chaque scan une fois.</b> Vers le dossier unique que nnUNet
lit, en NIfTI, par une vraie conversion SimpleITK et non un renommage.</p>
<p>Une fois, et pour toutes les structures : le checkpoint est alors chargé une
fois <i>par structure</i> et non une fois par (scan &times; structure), ce qui
sur une cohorte est la différence entre N&times;S et S chargements.</p>
</div></div>

<div class="step"><div class="n">4</div><div>
<p><b>Prédire, structure par structure, deux à la fois.</b> La largeur n'est
pas écrite dans le tool : il demande au serveur combien de structures il peut
mener de front, et le serveur répond avec ce que la place réservée pour ce run
peut payer.</p>
<pre><code>structure_width = min(sup.channels(len(models)), len(models), 2)</code></pre>
<p>Plafonné à deux : un troisième canal achète 7&nbsp;% de temps pour un
troisième modèle résident et ~1,2&nbsp;Gio de carte de plus. Une structure qui
échoue ne fait pas perdre les autres.</p>
</div></div>

<div class="step"><div class="n">5</div><div>
<p><b>Assembler.</b> Les masques sont repliés dans <b>l'ordre où les structures
ont été déclarées</b>, jamais dans celui où elles ont fini &mdash; le rapport
doit se lire pareil quel que soit l'ordonnancement. Forme fusionnée, séparée,
ou les deux.</p>
<p>Les surfaces (<code>.vtk</code>) sont optionnelles et décimées à 90&nbsp;%
par défaut : les <i>marching cubes</i> tournent sur la grille d'origine, donc
un CBCT à 0,33&nbsp;mm donne un triangle par face de voxel, pour une précision
que le masque n'a pas. 90 coûte 0,059&nbsp;mm d'écart moyen et divise la
géométrie par dix.</p>
</div></div>

<div class="step"><div class="n">6</div><div>
<p><b>Écrire.</b> Un dossier <code>&lt;scan&gt;_&lt;ID&gt;_SegOut/</code> par
scan, plus un <code>AMASSS_report.json</code> pour le lot. Rien n'est écrit
hors du dossier de sortie.</p>
<p>Le rapport porte ce qui a décidé du résultat &mdash; structures manquantes,
échecs par scan, <code>gpu_resampling</code>, <code>tile_step_size</code>,
<code>surface_decimation</code> &mdash; parce que ces sorties sont lossy par
défaut et qu'on doit pouvoir dire de combien.</p>
</div></div>

<h2>Où passe le temps</h2>

<p>Mesuré ici, sur un CBCT de 512&times;512&times;365 à 0,33&nbsp;mm, cinq
structures, avec une synchronisation CUDA à chaque frontière de phase. Les
secondes sont les <b>totaux sur les cinq structures</b>, pas une structure ; le
run entier fait 63,6&nbsp;s. Un run de contrôle sans les synchronisations donne
le même total, donc l'instrument n'a pas déformé ce qu'il mesure.</p>

<div class="wkey"><span><i class="wsw gpu"></i>sur la carte</span><span><i class="wsw host"></i>sur l'hôte, un cœur</span><span>la largeur de la barre est la part du run</span></div>
<table><tr><th>phase</th><th class="num">s</th><th>où ça tourne</th><th></th></tr>
<tr><td class="t">init</td><td class="num">4,1</td><td><div class="where"><div class="wbar" style="width:25px"><i style="width:0%"></i></div><span class="wlab">6&nbsp;% du run · carte 0,1&nbsp;%</span></div></td><td class="s">plans, réseau, checkpoint</td></tr>
<tr><td class="t">read</td><td class="num">4,3</td><td><div class="where"><div class="wbar" style="width:29px"><i style="width:0%"></i></div><span class="wlab">7&nbsp;% du run · carte 0,3&nbsp;%</span></div></td><td class="s">182&nbsp;Mio dégzippés du disque</td></tr>
<tr><td class="t">preprocess</td><td class="num">29,6</td><td><div class="where"><div class="wbar" style="width:197px"><i style="width:1%"></i></div><span class="wlab">47&nbsp;% du run · carte 0,9&nbsp;%</span></div></td><td class="s">normalisation et rééchantillonnage</td></tr>
<tr><td class="t">predict</td><td class="num">15,9</td><td><div class="where"><div class="wbar" style="width:105px"><i style="width:84%"></i></div><span class="wlab">25&nbsp;% du run · carte 83,7&nbsp;%</span></div></td><td class="s">la fenêtre glissante</td></tr>
<tr><td class="t">export</td><td class="num">8,1</td><td><div class="where"><div class="wbar" style="width:55px"><i style="width:16%"></i></div><span class="wlab">13&nbsp;% du run · carte 16,3&nbsp;%</span></div></td><td class="s">rééchantillonnage retour, argmax, écriture</td></tr>
</table>

<div class="note"><b>Un quart du run seulement met un kernel sur la carte.</b>
Le reste est un seul cœur occupé, ce qui est la signature d'un trou &mdash; pas
d'une machine saturée autrement. C'est ce trou que la largeur remplit.</div>

<h3>Les 23,6 s qui ne servaient à rien</h3>

<p>Dans <code>preprocess</code>, un seul appel &mdash;
<code>resample_seg</code> &mdash; pèse <b>23,6&nbsp;s</b>, soit plus du tiers
de la boucle, à 0,9&nbsp;% de GPU sur un cœur. <b>Et son résultat est jeté.</b>
Trois faits, vérifiés plutôt que supposés :</p>

<ul>
<li>le <code>seg</code> n'est la segmentation de personne : nnUNet le
<i>fabrique</i> deux lignes plus haut, c'est le masque des voxels non nuls ;</li>
<li>la normalisation le reçoit et ne le lit pas, et elle tourne
<b>avant</b> ce rééchantillonnage ;</li>
<li>son seul lecteur ensuite est derrière
<code>if folder_with_segs_from_prev_stage is not None</code>, le chemin en
cascade, que ces bundles n'utilisent pas.</li>
</ul>

<p>Le supprimer coûte <b>zéro voxel de différence</b>, Dice 1,00000000 sur les
cinq structures. Pour l'échelle : <code>resample_data</code>, celui pour lequel
le passage sur GPU avait été construit au prix d'un Dice de 0,978 sur la
vertèbre cervicale, pèse 0,7&nbsp;s.</p>

<h3>Deux structures à la fois</h3>

<p>La boucle d'inférence seule, même scan, cinq structures :</p>

<table>
<tr><th>largeur</th><th class="num">boucle</th><th class="num">gain</th>
    <th class="num">pic carte</th></tr>
<tr><td class="t">1</td><td class="num">62,6&nbsp;s</td><td class="num">&mdash;</td>
    <td class="num">5291&nbsp;Mio</td></tr>
<tr><td class="t">2</td><td class="num">38,6&nbsp;s</td><td class="num">1,62&times;</td>
    <td class="num">6371&nbsp;Mio</td></tr>
<tr><td class="t">3</td><td class="num">35,8&nbsp;s</td><td class="num">1,75&times;</td>
    <td class="num">7587&nbsp;Mio</td></tr>
</table>

<p>Les deux changements ensemble &mdash; le rééchantillonnage jeté qui
disparaît, puis deux structures de front &mdash; donnent, sur la passe de
couverture de cette machine, <b>76,8&nbsp;s &rarr; 35,6&nbsp;s, soit
2,16&times;</b>. À six clients simultanés : <b>275&nbsp;s &rarr;
154&nbsp;s</b>.</p>

<h2>Ce que ça coûte à la machine</h2>

<table>
<tr><th>run</th><th class="num">carte</th><th class="num">RAM hôte</th></tr>
<tr><td>une structure</td><td class="num">2,00&nbsp;Gio</td><td class="num">4,20&nbsp;Gio</td></tr>
<tr><td>cinq structures</td><td class="num">2,41&nbsp;Gio</td><td class="num">8,97&nbsp;Gio</td></tr>
</table>

<div class="note"><b>Un modèle résident à la fois.</b> C'est pourquoi cinq
structures coûtent à peine plus de carte qu'une seule, alors que la RAM hôte
double : elle accumule le volume de labels de chaque structure. Un second canal
ajoute environ 1&nbsp;Gio de carte, pas cinq.</div>

<h2>La reproductibilité, et ce qu'elle a coûté</h2>

<p>L'autotuning cuDNN est désormais <b>désactivé</b>, et celui-là déplace
vraiment des voxels : il coûte un Dice de 0,99992 sur la vertèbre cervicale
&mdash; à comparer aux 0,991 du rééchantillonnage GPU déjà actif par défaut,
environ 270 fois plus.</p>

<div class="note"><b>Ce qu'il achète : le même masque quelle que soit la charge
du serveur.</b> L'autotuning choisit un algorithme de convolution en
<i>chronométrant</i> des candidats. Avec deux structures sur la même carte les
chronos sont contendus, un autre algorithme gagne, et il arrondit autrement
&mdash; le même scan relancé rendait un masque différent. Les largeurs 1, 2 et
3 produisent maintenant des sorties identiques octet pour octet. La vitesse
n'est pas dans la balance : 38,6&nbsp;s sans autotuning contre 39,3&nbsp;s avec,
le même run au bruit près.</div>

<div class="note warnbox"><b><code>progress.set_width(1)</code> dans une boucle
qui en fait tourner deux : ce n'est pas un bug.</b> Le serveur apprend ce que
coûte un canal en divisant le pic d'un run par la largeur la plus étroite que
ce run a déclarée, et ce modèle n'a pas d'ordonnée à l'origine. Or AMASSS, sur
cette machine, c'est 5291&nbsp;Mio à la largeur 1, 6371 à 2 et 7587 à 3 : ~4,1
Gio <b>fixes</b> plus ~1,15&nbsp;Gio par structure. Déclarer la vérité
enseignerait <code>6371&nbsp;/&nbsp;2&nbsp;=&nbsp;3185</code>&nbsp;Mio par
canal, contre 5291 qu'un prochain run en largeur 1 réclame &mdash; 40&nbsp;%
sous-réservé, et sous-réserver est la direction de l'<i>out of memory</i>. Des
deux erreurs que le modèle impose, une seule est sûre. Le prix est réel et
assumé : à la largeur 2, ce run réserve ~10,4&nbsp;Gio pour en utiliser
~6,4.</div>

<div class="note alertbox"><b>Connu, ouvert :</b> dans le préprocessing qui
reste, <code>crop</code> pèse 4,2&nbsp;s et <code>init</code> 4,1&nbsp;s, et
personne n'a encore ouvert ni l'un ni l'autre. Ce sont les deux plus gros blocs
mono-cœur qui subsistent après le rééchantillonnage jeté.</div>

<h2>Arguments</h2>

<table>
<tr><th>nom</th><th>type</th><th>défaut</th><th></th></tr>
<tr><td class="t">scans</td><td class="t">chemin</td><td>&mdash;</td>
    <td>un CBCT orienté, ou un dossier pour un lot</td></tr>
<tr><td class="t">model</td><td class="t">nom</td><td>&mdash;</td>
    <td>un bundle détenu par ce serveur, un sous-dossier par structure</td></tr>
<tr><td class="t">structures</td><td class="t">multichoix</td>
    <td class="t">MAND MAX CB CV UAW</td>
    <td>9 codes ; celle dont le bundle n'a pas les poids est signalée</td></tr>
<tr><td class="t">merge</td><td class="t">multichoix</td><td class="t">MERGED</td>
    <td>fusionnée, séparée, ou les deux</td></tr>
<tr><td class="t">prediction_ID</td><td class="t">str</td><td class="t">Pred</td>
    <td>suffixe des noms de sortie</td></tr>
<tr><td class="t">generate_surface</td><td class="t">bool</td><td class="t">false</td>
    <td>exporte aussi une surface par masque</td></tr>
<tr><td class="t">surface_smoothing</td><td class="t">int</td><td class="num">5</td>
    <td>technique ; sans effet sans les surfaces</td></tr>
<tr><td class="t">surface_decimation</td><td class="t">int</td><td class="num">90</td>
    <td>technique ; 0 garde le maillage brut</td></tr>
<tr><td class="t">device</td><td class="t">cuda | cpu</td><td class="t">cuda</td>
    <td>technique ; bascule sur CPU avec un avertissement si aucune carte</td></tr>
<tr><td class="t">tile_step_size</td><td class="t">float</td><td class="num">0.5</td>
    <td>technique ; il déplace la segmentation, donc laissé au défaut nnUNet</td></tr>
<tr><td class="t">gpu_resampling</td><td class="t">bool</td><td class="t">true</td>
    <td>technique ; false rend une sortie identique à nnUNet, bit à bit</td></tr>
<tr><td class="t">num_workers</td><td class="t">int</td><td class="num">1</td>
    <td>technique ; scans lus et écrits de front, ne touche pas l'inférence</td></tr>
</table>

<p>Les lignes marquées « technique » ne sont pas présentées au clinicien : le
tool les déclare, le serveur les remplit.</p>

<footer>Mesuré sur ce déploiement, sur une carte RTX 6000 Ada. Les chiffres
viennent du profil par phase, de la passe de couverture et de la campagne de
charge, pas d'une spécification.</footer>
"""

_ALI_CBCT = """
<div class="crumb">Doc</div>
<h1>ALI_CBCT &mdash; identification automatique des landmarks</h1>
<div class="sub">Un agent par point anatomique, qui marche dans le volume
jusqu'à s'arrêter dessus.</div>
</header>

<p>ALI_CBCT place des points anatomiques sur un CBCT. Pas de segmentation, pas
de carte de chaleur : <b>un agent par landmark</b>, entraîné par apprentissage
par renforcement, part d'une position et fait un pas dans l'une de six
directions, d'abord à 1&nbsp;mm puis à 0,3&nbsp;mm, jusqu'à converger. Le
catalogue compte <b>119 landmarks</b> ; le bundle porte un réseau par landmark
et par échelle, 56&nbsp;Mo chacun, 13&nbsp;Go en tout.</p>

<h2>Le déroulé d'un run</h2>

<div class="step"><div class="n">1</div><div>
<p><b>Reconnaître l'entrée.</b> Un volume, un dossier parcouru récursivement,
ou une série DICOM &mdash; convertie automatiquement, dans le répertoire de
travail de la requête et jamais à côté des données du client.</p>
<p>Une surface intra-orale est <b>refusée en nommant le fichier et le tool à
lancer à la place</b>, plutôt qu'à moitié traitée ; une entrée qui mélange
volumes et surfaces demande deux lots.</p>
</div></div>

<div class="step"><div class="n">2</div><div>
<p><b>Préparer le scan.</b> Correction d'histogramme &mdash; les agents ont
été entraînés sur des volumes normalisés ainsi, c'est le contrat du modèle et
non un ornement &mdash; puis rééchantillonnage aux deux échelles.</p>
<pre><code>SCALE_SPACINGS = (1.0, 0.3)      # &lt;landmark&gt;/1/ et &lt;landmark&gt;/0-3/</code></pre>
<p><b>12 à 13&nbsp;s, strictement séquentielles</b>, pendant lesquelles aucun
canal n'existe encore.</p>
</div></div>

<div class="step"><div class="n">3</div><div>
<p><b>Choisir les agents.</b> Par régions, ou par points nommés &mdash; et
nommer des points <b>REMPLACE</b> la sélection par régions au lieu de la
restreindre.</p>
<pre><code>regions   = ["Cranial base", "Upper", "Lower", "Impacted canine"]
landmarks = []           # vide : ce sont les régions qui décident</code></pre>
<p>C'est ce qui permet de demander les sept points dont on a besoin au lieu de
faire tourner 58 agents pour en utiliser sept. Un landmark dont le bundle n'a
pas les poids coûte une ligne du rapport, pas le run.</p>
</div></div>

<div class="step"><div class="n">4</div><div>
<p><b>Chercher.</b> Chaque agent a son propre checkpoint, son propre
générateur, et une borne en <b>nombre de pas</b>.</p>
<pre><code>MAX_STEPS_PER_SCALE = 2000       # ~4,5x la pire marche qui converge
_MAX_ATTEMPTS       = 3          # sorties du volume avant abandon</code></pre>
<p>Une borne en pas et non en horloge : c'est le même travail pour tout le
monde, donc le même scan avec les mêmes poids et la même graine place les
mêmes points sur une machine chargée que sur une machine libre. Borné par une
horloge, il ne le faisait pas &mdash; une carte partagée à quatre coûtait au
scan de référence un landmark que deux trouvaient.</p>
</div></div>

<div class="step"><div class="n">5</div><div>
<p><b>Ouvrir la largeur.</b> L'unité parallélisable est le <i>landmark</i>, pas
le scan, et ce compte n'est connu qu'ici : c'est ce que le bundle a en poids
croisé avec ce qui a été demandé.</p>
<pre><code>width_from = "landmarks"
workers    = min(sup.channels(len(runnable)), len(runnable))</code></pre>
<p>Un canal est un <b>processus</b>, pas un thread : un pas, c'est quelques
microsecondes de kernel autour de beaucoup de Python, et le GIL sérialisait les
marches. Le résultat ne dépend pas de la largeur &mdash; le fichier de 119
landmarks à huit canaux a le même sha256 qu'à un.</p>
</div></div>

<div class="step"><div class="n">6</div><div>
<p><b>Écrire.</b> Un <code>&lt;scan&gt;_lm_&lt;ID&gt;.mrk.json</code> par scan,
l'arborescence du dossier d'entrée reproduite, plus un
<code>run_report.json</code>.</p>
<p>Le rapport se lit dans l'ordre demandé quel que soit l'ordre dans lequel les
recherches ont fini, et il nomme les landmarks non placés avec la raison.</p>
</div></div>

<h2>Où passe le temps</h2>

<p>Un pas, instrumenté avec des synchronisations CUDA. Un landmark en fait de
l'ordre de 150.</p>

<table>
<tr><th>dans un pas</th><th class="num">ms</th><th class="num">part</th></tr>
<tr><td>crop + rescale + cast + hôte&rarr;carte</td><td class="num">1,2</td>
    <td class="num">7&nbsp;%</td></tr>
<tr><td>passe avant du DenseNet</td><td class="num">15,2</td>
    <td class="num">92&nbsp;%</td></tr>
<tr><td>argmax</td><td class="num">0,06</td><td class="num">&mdash;</td></tr>
</table>

<div class="note"><b>Les transformations n'ont jamais été le temps. Ce
qu'elles coûtaient, c'étaient des cœurs.</b> Une opération torch sur
262&nbsp;144 éléments entre dans le pool intra-op &mdash; 28 threads ici
&mdash; et libgomp continue de tourner après une région parallèle de
0,6&nbsp;ms : un worker occupait ainsi ~10 des 56 cœurs de la machine.
Déplacer le volume rééchantillonné sur la carte, une ligne, ramène cela à ~1
cœur &mdash; et transforme la largeur 8, jusque-là la configuration la plus
<i>lente</i>, en la plus rapide.</div>

<h2>L'échelle : ce que coûte un landmark de plus</h2>

<p>Seul sur la machine, un scan, la largeur décidée par le serveur :</p>

<table>
<tr><th>landmarks demandés</th><th class="num">secondes</th>
    <th class="num">canaux</th><th class="num">pic carte</th></tr>
<tr><td class="t">1</td><td class="num">21,2</td><td class="num">1</td>
    <td class="num">0,62 Gio</td></tr>
<tr><td class="t">3</td><td class="num">27,8</td><td class="num">3</td>
    <td class="num">3,44 Gio</td></tr>
<tr><td class="t">7</td><td class="num">32,0</td><td class="num">7</td>
    <td class="num">7,80 Gio</td></tr>
<tr><td class="t">15</td><td class="num">50</td><td class="num">15</td>
    <td class="num">16,74 Gio</td></tr>
<tr><td class="t">30</td><td class="num">55</td><td class="num">15</td>
    <td class="num">16,92 Gio</td></tr>
<tr><td class="t">60</td><td class="num">87</td><td class="num">15</td>
    <td class="num">16,89 Gio</td></tr>
<tr><td class="t">119</td><td class="num">144</td><td class="num">15</td>
    <td class="num">16,92 Gio</td></tr>
</table>

<p>Soit, en gros, <b>25&nbsp;s de coût fixe plus 1&nbsp;s par landmark</b>
au-delà de 30. Ce matin, le landmark marginal en coûtait 4,4. La requête par
défaut &mdash; les quatre régions, 58 agents &mdash; est passée de
<b>476&nbsp;s à 87&nbsp;s, 5,5&times;</b>, avec un sha256 identique à toutes
les largeurs : les 115 points n'ont pas bougé.</p>

<div class="note warnbox"><b>Et le mur, mesuré deux fois par deux chemins.</b>
Les 119 landmarks prennent <b>137,8&nbsp;s à une largeur de 8</b> et
<b>144&nbsp;s à 15</b>. Deux fois plus de canaux, et plus lent : la carte
sature autour de huit agents concurrents, donc au-delà la largeur achète de la
mémoire et pas de la vitesse. Le coût par canal, lui, est linéaire et mesuré :
0,78&nbsp;Gio de carte et ~1,5&nbsp;Gio d'hôte, au-dessus de ~0,7&nbsp;Gio que
le run tient de toute façon.</div>

<div class="note alertbox"><b>Connu, ouvert :</b> les ~25&nbsp;s de coût fixe
n'ont pas été ouvertes. On en connaît 12 à 13 &mdash; la correction
d'histogramme et les deux rééchantillonnages, séquentiels par construction
puisqu'aucun canal n'existe encore &mdash; et le reste n'est pas attribué.
Au-delà, la seule route restante vers un temps plat est de <b>grouper les
passes avant</b> : 92&nbsp;% d'un pas est un DenseNet sur un seul cube de
64&sup3;, et rien aujourd'hui ne met les agents d'un même pas dans le même
batch.</div>

<h2>Arguments</h2>

<table>
<tr><th>nom</th><th>type</th><th>défaut</th><th></th></tr>
<tr><td class="t">input</td><td class="t">chemin</td><td>&mdash;</td>
    <td>un CBCT, un dossier, ou une série DICOM</td></tr>
<tr><td class="t">model</td><td class="t">nom</td><td>(le sien)</td>
    <td>vide : le bundle que ce serveur héberge pour ce tool</td></tr>
<tr><td class="t">regions</td><td class="t">multichoix</td>
    <td class="t">les quatre</td>
    <td>Cranial base, Upper, Lower, Impacted canine</td></tr>
<tr><td class="t">landmarks</td><td class="t">multichoix</td><td>(vide)</td>
    <td>nommer un point REMPLACE la sélection par régions</td></tr>
<tr><td class="t">prediction_ID</td><td class="t">str</td><td class="t">Pred</td>
    <td>suffixe des noms de sortie</td></tr>
<tr><td class="t">device</td><td class="t">cuda | cpu</td><td class="t">cuda</td>
    <td>technique ; bascule sur CPU avec un avertissement si aucune carte</td></tr>
<tr><td class="t">search_steps</td><td class="t">int</td><td class="num">0</td>
    <td>technique ; garde-fou compt&eacute; en passes avant, pas en secondes
        &mdash; 0 vaut 900, l&rsquo;agent le plus lent mesur&eacute; en
        prenant 193. Le m&ecirc;me nombre sur toute machine&nbsp;: c&rsquo;est
        ce qui permet d&rsquo;&eacute;largir le run sans qu&rsquo;un point
        cesse d&rsquo;&ecirc;tre trouv&eacute;</td></tr>
<tr><td class="t">seed</td><td class="t">int</td><td class="num">0</td>
    <td>technique ; la graine de respawn, donc la reproductibilité</td></tr>
<tr><td class="t">num_workers</td><td class="t">int</td><td class="num">1</td>
    <td>technique ; la largeur hors serveur seulement &mdash; sous serveur,
        c'est le serveur qui répond</td></tr>
</table>

<p>Les lignes marquées « technique » ne sont pas présentées au clinicien : le
tool les déclare, le serveur les remplit.</p>

<footer>Mesuré sur ce déploiement, sur une carte RTX 6000 Ada, avec le bundle
réel. Un run qui finit sur le garde-fou d'horloge est un défaut à signaler,
pas un scan difficile.</footer>
"""

DOCS = {"CNE": _CNE, "AMASSS": _AMASSS, "ALI_CBCT": _ALI_CBCT}


def page_for(tool: str) -> str:
    """The document for one tool, or "" when nobody has written it."""
    body = DOCS.get(tool)
    if not body:
        return ""
    return (
        "<!doctype html><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{tool} \u2014 doc</title>"
        f"<style>{_STYLE}</style>"
        f"<div class='wrap'><header>{body}</div>"
    )
