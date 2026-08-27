# Relatório de recomendação de investimento — Itapema

## 1. Executive summary

Recomendamos que a Seazone compre **um apartamento de 2 quartos em Meia Praia (Itapema, SC)**. A escolha parte da **convergência entre múltiplas perspectivas complementares**: performance estática (static revenue proxy H91 ≈ R$15.925; capital efficiency ≈ 1,47%), dinâmica temporal (pickup mediano ≈ R$1.541,92) e robustez amostral (126 listings / 112 owners, faixa amostral forte apesar de cobertura Price_AV snapshot de 17,4%, Pareto não dominado). Meia Praia não tem a maior eficiência isolada — Morretes 2Q tem ~1,60% e preço menor — mas é o perfil mais defensável para uma compra hoje.

Todas as métricas são **proxies**; nunca receita ou ocupação realizada. Nada é anualizado.

## 2. Problema

O desafio pede uma recomendação de investimento imobiliário para a Seazone em Itapema (SC), respondendo a quatro perguntas: melhor perfil, melhor localização, drivers de receita e decisão de compra. Também pede tomar posição sobre a tese interna de que "studio/1 quarto no Centro" seria a alternativa mais eficiente.

## 3. Dados

Cinco datasets (snapshot estático):
- `Details_Itapema.csv` — listings Airbnb;
- `Hosts_ids_Itapema.csv` — anfitriões;
- `Mesh_Ids_Data_Itapema.csv` — geolocalização/bairro;
- `Price_AV_Itapema.csv` — preço por listing × data;
- `VivaReal_Itapema.csv` — mercado de compra.

## 4. Auditoria e qualidade dos dados

- `Details` e `Mesh` são únicos por `airbnb_listing_id`.
- `Hosts` tem múltiplas linhas por `owner_id`; usado último snapshot por owner (3.057 owners).
- `Price_AV`: 3 capture_day (06/07/20 de jan/2025), cada um cobrindo D..D+90 (91 datas); sem duplicação (listing, capture_day, stay).
- `VivaReal`: 8.293 IDs únicos após deduplicação; bairros canonicalizados.
- Ratings iguais a zero em listings sem reviews foram tratados como missing.

## 5. Interpretação do Price_AV

- Linha presente = data com preço anunciado.
- Ausência dentro da janela para listing observado = indisponibilidade (proxy).
- Indisponibilidade **não** é reserva; pode ser bloqueio, manutenção ou uso do proprietário.
- Essa interpretação é uma inferência semântica e foi tratada explicitamente como risco no red-team.

## 6. Construção das métricas

- **static_revenue_proxy_H** = `median_available_price_H × unavailable_nights_H`.
- **capital_efficiency_proxy** = `static_revenue_proxy / median_asking_sale_price`.
- **pickup** = mudança de disponível → indisponível entre capturas, valorizada pelo preço.
- Horizontes H30/H60/H91; snapshot principal 20/01.

## 7. Melhor perfil

**Apartamento de 4 quartos** — N=45, static revenue proxy H91 mediana ≈ R$31.941, ADR ≈ R$874. É o perfil de maior receita absoluta, mas **não** o melhor investimento (eficiência de capital favorece 2Q). Segmentos 5Q/6Q/12Q têm N muito pequeno e não são robustos.

## 8. Melhor localização

**Meia Praia** para static revenue proxy (H30 ≈ R$12.874; H60 ≈ R$18.850; H91 ≈ R$19.200). O **Centro** tem a maior intensidade de pickup mediano por listing (momentum). Não confundir bairro Centro com o segmento Centro 2Q.

## 9. Drivers de performance

Associações (descritivas):
- ADR: quartos, banheiros, hóspedes, cleaning fee, camas.
- Unavailability proxy: reviews, fotos, maturidade do host.
- Static revenue: principalmente estrutura/capacidade, associada ao ADR e à proxy estática observada.

GroupKFold por owner mostrou que o Random Forest **não generaliza** static revenue para hosts não vistos ⇒ os drivers representam associações descritivas, não relações causais nem evidência preditiva forte.

## 10. Mercado de compra / VivaReal

VivaReal deduplicado (8.293 IDs), bairros canonicalizados, preços medianos por segmento. Preços são **asking price**, não transações.

## 11. Eficiência de capital

- Meia Praia 2Q ≈ 1,47%
- Morretes 2Q ≈ 1,60%
- Centro 2Q ≈ 1,30%
- Centro 1Q ≈ 1,17%

## 12. Sensibilidade H30/H60/H91

Eficiência observada por horizonte; os rankings mudam pouco entre H30/H60/H91 (Morretes 1º/2º/1º, Meia 2º/1º/2º, Centro 3º, Centro1 4º).

## 13. Bootstrap e incerteza

Bootstrap cluster por owner (2000 reps). Morretes 2Q mediana ≈1,61% (P2.5–97.5 1,36–2,05); Meia 2Q ≈1,48% (1,28–1,67). Os pairwise win rates do bootstrap favorecem a mesma ordenação em várias comparações (ex.: Morretes > Meia em aproximadamente 82% das reamostragens do bootstrap principal), mas medem estabilidade amostral condicional aos dados observados, não probabilidades econômicas verdadeiras.

## 14. Pickup / dinâmica temporal

Separamos dois níveis de granularidade: **bairro** e **segmento**.

**Por bairro** (pickup mediano por listing entre os principais bairros):

- Centro ≈ R$3.365
- Meia Praia ≈ R$2.397
- Morretes ≈ R$0

**Por segmento** (pickup mediano por segmento específico):

- Centro 2Q ≈ R$3.982,58
- Centro 1Q ≈ R$3.469,96
- Meia Praia 2Q ≈ R$1.541,92
- Morretes 2Q ≈ R$0

A conclusão por bairro (Centro com maior intensidade de pickup) e os números por segmento referem-se a granularidades diferentes e não devem ser lidos como equivalentes.

## 15. Pareto e trade-offs

Fronteira entre capital efficiency × pickup. Não-dominados: Morretes 2Q, Meia Praia 2Q, Centro 2Q. Centro 1Q é dominado. Sem score ponderado.

## 16. Red-team

10 riscos documentados em `outputs/tables/redteam_risks.csv` (4 críticos, 5 importantes, 1 secundário). Principais: semântica Price_AV, indisponível≠reserva, selection bias, sazonalidade, asking price, custos não observados, dependência por host, amostra pequena, construção da static proxy.

## 17. Tese dos compactos no Centro

- **Studio**: inconclusivo por evidência insuficiente (3 listings, 0 com preço).
- **Centro 1Q**: não sustentado como maior eficiência de capital (4º nos horizontes, cap_eff 1,17%, Pareto dominado, amostra venda pequena). A favor: cobertura 64,7% e pickup R$3.469,96.

## 18. Recomendação final

**PRIMARY: Apartamento 2Q em Meia Praia.** Alternativas: **Morretes 2Q** (eficiência/value) e **Centro 2Q** (momentum). Decisão por convergência estático + dinâmico + robustez amostral.

## 19. Limitações

Semântica Price_AV inferida; indisponibilidade≠reserva; selection bias/coverage heterogênea (Meia 17,4%); janela sazonal (sem anualização); asking price≠transação; custos operacionais completos não observados; Airbnb≠VivaReal imóvel-a-imóvel; dependência por host; amostras desiguais; static revenue é proxy.

## 20. Reprodutibilidade

Ambiente: `python -m venv .venv`, `pip install -r requirements.txt`. Execução dos scripts `01..06` + `99_validate_final_artifacts.py`. `data/` não é modificado.

## 21. Uso de IA

- OpenCode: implementação/execução dos scripts.
- ChatGPT: metodologia, auditoria independente, red-team.
- Humano: decisão, crítica, revisão e aprovação.

Erros metodológicos e de implementação foram detectados e corrigidos. Logs completos serão em `ai-log/`. A decisão final é humana.