# Identifikation af RNA-modifikationer i polyA-haler med Nanopore-sekventering (og machine learning)

***OBS***: *Hold gerne Ctrl/Cmd nede, når der klikkes på links, så de åbner i en ny fane.*

### Vejledning til læsning af Github:
**1. Start med at læse projektbeskrivelsen**: <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/Projektbeskrivelse.pdf"><code>Projektbeskrivelse.pdf</code></a>

**2. Mappe: 02 Data Visualization**
   - Kig først på plots af de rå polyA-hale signaler i mappen 'Raw PolyA Signal', som viser et udsnit råsignaler fra data med 60 baser med henholdsvis én modifikation og ingen modifikation:
        - <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/02%20Data%20Visualization/Raw%20PolyA%20Signal/raw_polyA_signals_a60_60.pdf"><code>raw_polyA_signals_a60_60.pdf</code></a>
        - <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/02%20Data%20Visualization/Raw%20PolyA%20Signal/raw_polyA_signals_a60_unmod.pdf"><code>raw_polyA_signals_a60_unmod.pdf</code></a>
   - Kig derefter på plots af længderne af polyA-halerne i mappen 'PolyATail Length':
        - <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/02%20Data%20Visualization/PolyATail%20Length/PolyATail%20Lengths%20Plots.pdf"><code>PolyATail Lengths Plots.pdf</code></a>

   *Formålet med dette er for vise, hvor varierende rådataen er - både i længde og støj.*
 
**3. Mappe: 03 Vectorization**
   - Kig på <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/03%20Vectorization/Vectorization_median_plots.pdf"><code>Vectorization_median_plots.pdf</code></a>, hvor der er lavet vektorisering på data. (Kig sæligt på A60_60 og A60_unmod)
 
*Dette er for at vise vores første forsøg på at gøre rådataen mere ensartet; ved at lave vektorer af ens længde (vektorisering) vha. medianen og se om, der var tydelige ændringer i gennemsnitet og variansen på tværs af hele datasættet for de forskellige længder og modifikationsantal.*

**4. Mappe: 04 Moving Vectorization**
   - Se plots af moving vectorization (**rullende vektorisering??**) for datasætene med henholdsvis én modifikation og uden modifikation for både gennemsnit og varians:
      - <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/04%20Moving%20Vectorization/a60_60_mean_plots.pdf"><code>a60_60_mean_plots.pdf</code></a>
      - <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/04%20Moving%20Vectorization/a60_60_var_plots.pdf"><code>a60_60_var_plots.pdf</code></a>
      - <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/04%20Moving%20Vectorization/a60_unmod_mean_plots.pdf"><code>a60_unmod_mean_plots.pdf</code></a>
      - <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/04%20Moving%20Vectorization/a60_unmod_var_plots.pdf"><code>a60_unmod_var_plots.pdf</code></a>
   - Denne fil er den samme som de øverste fire, men blot for ét enkelt read i stedet for gennemsnitet af hele datasættet:
        - <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/04%20Moving%20Vectorization/Outliers%20fjernelse%20og%20plot%20af%20%C3%A9t%20signal.ipynb"><code>Outliers fjernelse og plot af ét signal.ipynb</code></a>

*Dette er den metode af vektorisering, vi endte med at gå videre med - moving vectorization på et enkelt dataread ad gangen. Kort fortalt: Den laver den nye vektor ved at rulle et vindue hen over dataen og tage gennemsnit af det i vinduet og tilføje det til den nye vektor. Dette er gjort for både gennemsnittet og variansen igen.*
 
**5. Mappe: 05 Shiny App**
   - Skærmbillede af vores Shiny Appen (som eksempel): <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/05%20Shiny%20App/Shiny%20App%20Eksempel.pdf"><code>Shiny App Eksempel.pdf</code></a>
   - <a href="https://juliemalm.github.io/Video-dataprojekt/"><code>Video af Shiny App i brug</code></a>
   - Link til app koden: <a href="https://github.com/Najaandrup/Dataprojekt/blob/main/05%20Shiny%20App/app.py"><code>app.py</code></a>

     *Dette er vores shiny app, som bruger metoden, moving vectorization, fra ovenover. Man kan altså uploade sine datasæt med og uden modifikationer og sammenligner dem vha. moving vectorization.*

**Prøv eventuelt selv appen**:
   - Åben appen vha. linket: <a href="https://naja.shinyapps.io/05_shiny_app/"><code>Shiny App</code></a>
   - Download de fire datafiler her: <a href="https://github.com/Najaandrup/Dataprojekt/tree/main/05%20Shiny%20App/Data%20for%20Shiny%20App"><code>Data for Shiny App</code></a>
   - Upload dem i appen og test funktionaliteten.

     *(Data der hedder a60_60 er med modifikation, de skal uploades til venstre og a_60_unmod skal uploades til højre)*








