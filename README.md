# Identifikation af RNA-modifikationer i polyA-haler med Nanopore-sekventering (og machine learning)

**OBS**: *Hold gerne `Ctrl`/`Cmd` (Windows/Mac) nede, når der klikkes på link, således de åbnes i en ny fane.*

### Vejledning til læsning af Github:
- Læs <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/Projektbeskrivelse.pdf"><code>Projektbeskrivelse.pdf</code></a>

-  Mappen: 02 Data Visualization
    - Raw PolyA Signal: <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/02%20Data%20Visualization/Raw%20PolyA%20Signal/raw_polyA_signals_a60_60.pdf"><code>raw_polyA_signals_a60_60.pdf</code></a> & <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/02%20Data%20Visualization/Raw%20PolyA%20Signal/raw_polyA_signals_a60_unmod.pdf"><code>raw_polyA_signals_a60_unmod.pdf</code></a>
    - PolyATail Length: <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/02%20Data%20Visualization/PolyATail%20Length/PolyATail%20Lengths%20Plots.pdf"><code>PolyATail Lengths Plots.pdf</code></a>

     *Dette er for at få en fornemmelse af, hvor varierende rådataen er - både i støj og længde.*
 
- Mappen: 03 Vectorization
    - <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/03%20Vectorization/Vectorization_median_plots.pdf"><code>Vectorization_median_plots.pdf</code></a> (kig på a60_60 og a60_unmod)
 
     *Dette er for at vise vores første forsøg på at gøre rådataen mere ensartet; ved at lave vektorer af ens længde (vektorisering) vha. medianen og se om, der var tydelige ændringer i gennemsnitet og variansen på tværs af hele datasættet for de forskellige længder og modifikationsantal.*

- Mappen: 04 Moving Vectorization
    - <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/04%20Moving%20Vectorization/a60_60_mean_plots.pdf"><code>a60_60_mean_plots.pdf</code></a>
    - <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/04%20Moving%20Vectorization/a60_60_var_plots.pdf"><code>a60_60_var_plots.pdf</code></a>
    - <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/04%20Moving%20Vectorization/a60_unmod_mean_plots.pdf"><code>a60_unmod_mean_plots.pdf</code></a>
    - <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/04%20Moving%20Vectorization/a60_unmod_var_plots.pdf"><code>a60_unmod_var_plots.pdf</code></a>
    - <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/04%20Moving%20Vectorization/Outliers%20fjernelse%20og%20plot%20af%20%C3%A9t%20signal.ipynb"><code>Outliers fjernelse og plot af ét signal.ipynb</code></a>
        - (*Den nederste fil er den samme som de øverste fire, blot for et enkelt read i stedet for gennemsnitet af hele datasættet.*)

     *Dette er den metode af vektorisering, vi endte med at gå videre med - moving vectorization på et enkelt dataread ad gangen. Kort fortalt: Den laver den nye vektor ved at rulle et vindue hen over dataen og tage gennemsnit af det i vinduet og tilføje det til den nye vektor. Dette er gjort for både gennemsnittet og variansen igen.*
 
- Mappen: 05 Shiny App
    - <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/05%20Shiny%20App/Shiny%20App%20Eksempel.pdf"><code>Shiny App Eksempel.pdf</code></a>
    - <a href="https://juliemalm.github.io/Video-dataprojekt/"><code>Video af Shiny App i brug</code></a>
    - Link til app kode: <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/05%20Shiny%20App/app.py"><code>app.py</code></a>

     *Dette er vores shiny app, som bruger metoden fra ovenover - så kan man uploade sine datasæt med og uden modifikationer og sammenligner dem vha. moving vectorization.*

Hvis tid, så kan appen afprøves:
- Link til app: <a href="https://naja.shinyapps.io/05_shiny_app/"><code>Shiny App</code></a>
- Download de fire filer under <a href="https://github.com/Najaandrup/Dataprojekt/tree/main/05%20Shiny%20App/Data%20for%20Shiny%20App"><code>Data for Shiny App</code></a> og brug dem i appen








