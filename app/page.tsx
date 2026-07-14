const repoUrl = "https://github.com/thewrik/pm-volatility-extension";

const metrics = [
  { value: "1,007,560", label: "hourly out-of-sample forecasts" },
  { value: "1,404", label: "prediction-market contracts" },
  { value: "50", label: "test months" },
  { value: "0.1667", label: "DR-AS interval-score gain over deadline-only" },
];

const evidence = [
  {
    eyebrow: "What held up",
    title: "Active updates matter.",
    body: "The replication finds that adding active-search pressure to deadline pressure substantially improves interval forecasts when prices actually move.",
  },
  {
    eyebrow: "What the new model adds",
    title: "No change becomes part of the forecast.",
    body: "The hurdle-beta model forecasts both the chance that nothing happens and the size of the move if something does happen, while preserving the martingale mean.",
  },
  {
    eyebrow: "What did not win",
    title: "Coherence is not a magic wand.",
    body: "MHB beats two natural ablations on tick-aware log score, but it does not dominate the active-fit DR-AS normal benchmark on the primary scores.",
  },
];

export default function Home() {
  return (
    <main>
      <section className="hero">
        <nav className="topbar" aria-label="Primary links">
          <a href={repoUrl}>Repository</a>
          <a href="https://arxiv.org/abs/2607.08199">Original paper</a>
          <a href="/when-nothing-happens.pdf">PDF</a>
        </nav>

        <div className="heroGrid">
          <div className="heroCopy">
            <p className="kicker">A layperson-friendly research exhibit</p>
            <h1>When nothing happens, prediction markets are still telling you something.</h1>
            <p className="lede">
              This project extends a 2026 prediction-market volatility paper by asking
              a simple missing question: what if the next best forecast is that the
              price will not move at all?
            </p>
            <div className="actions">
              <a className="primaryAction" href={repoUrl}>
                View repository
              </a>
              <a className="secondaryAction" href="#plain-english">
                Read the plain-English version
              </a>
            </div>
          </div>

          <aside className="heroPanel" aria-label="Research status">
            <span className="statusDot" />
            <p className="panelLabel">Status</p>
            <h2>Promising, bounded, and deliberately not oversold.</h2>
            <p>
              The theorem is exact under its assumptions. The empirical result is
              mixed. The novelty and accuracy of model-generated outputs remain
              subject to independent verification.
            </p>
          </aside>
        </div>
      </section>

      <section className="notice" aria-label="Verification notice">
        <p>
          This repository is a one-shotting exercise for <strong>5.6-Sol</strong>.
          It is intended as an auditable research artifact, not a final peer-reviewed
          claim. Accuracy, correctness, and novelty of model outputs should be
          independently verified before citation or operational use.
        </p>
      </section>

      <section className="metrics" aria-label="Backtest scale">
        {metrics.map((metric) => (
          <div key={metric.label} className="metric">
            <strong>{metric.value}</strong>
            <span>{metric.label}</span>
          </div>
        ))}
      </section>

      <section id="plain-english" className="section twoColumn">
        <div>
          <p className="kicker">Plain-English idea</p>
          <h2>Most volatility models focus on the jump. This one also models the wait.</h2>
        </div>
        <div className="prose">
          <p>
            A prediction-market price is like a live probability. If the market says
            an event has a 60% chance, the price is around 60 cents. Existing models
            can describe how big the next move might be when traders update the price.
          </p>
          <p>
            But prediction markets often sit still. A price can remain unchanged not
            because nothing is happening in the world, but because no new public or
            private information has crossed the threshold needed to move the market.
            This project turns that stillness into a first-class forecast target.
          </p>
        </div>
      </section>

      <section className="section evidenceGrid" aria-label="Evidence summary">
        {evidence.map((item) => (
          <article key={item.title} className="evidenceCard">
            <p>{item.eyebrow}</p>
            <h3>{item.title}</h3>
            <span>{item.body}</span>
          </article>
        ))}
      </section>

      <section className="section modelBlock">
        <div className="modelCopy">
          <p className="kicker">The technical contribution</p>
          <h2>A bounded-martingale rule that keeps the forecast honest.</h2>
          <p>
            If a market price is a probability, it cannot wander outside zero and one.
            That simple boundary forces a useful restriction: normalized variance
            release cannot exceed the probability of a real price update.
          </p>
        </div>
        <div className="formulaCard" aria-label="Main theorem inequality">
          <p>Bounded martingale price</p>
          <strong>0 &lt;= r &lt;= q &lt;= 1</strong>
          <span>
            r is normalized variance release. q is the probability that the price
            changes. The new hurdle-beta distribution is built to obey this exactly.
          </span>
        </div>
      </section>

      <section className="section chartSection">
        <div>
          <p className="kicker">Audit trail</p>
          <h2>The repo includes tests, simulations, and a frozen public-data backtest.</h2>
          <p>
            The chart below is one of the simulation diagnostics kept with the
            artifact. The stronger audit trail is in the repository: tests, locked
            results, claims ledger, novelty notes, and the manuscript source.
          </p>
        </div>
        <img
          src="/hazard-reliability.png"
          alt="Simulation diagnostic showing hazard reliability for the model"
        />
      </section>

      <section className="section linksPanel" aria-label="Project links">
        <div>
          <p className="kicker">Read next</p>
          <h2>Follow the artifact, not just the story.</h2>
        </div>
        <div className="linkList">
          <a href={repoUrl}>
            <span>Repository</span>
            <strong>Code, tests, data manifest, and manuscript source</strong>
          </a>
          <a href="/when-nothing-happens.pdf">
            <span>Paper PDF</span>
            <strong>Eight-page manuscript version built from the repo</strong>
          </a>
          <a href={repoUrl + "/blob/main/docs/claims.md"}>
            <span>Claims ledger</span>
            <strong>What is supported, rejected, or only descriptive</strong>
          </a>
          <a href={repoUrl + "/blob/main/docs/novelty.md"}>
            <span>Novelty notes</span>
            <strong>Closest prior work screened and remaining verification risk</strong>
          </a>
        </div>
      </section>
    </main>
  );
}
