import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import Heading from '@theme/Heading';
import styles from './index.module.css';
import AppScreenshot from '@site/static/img/violencedetectionsystem_app.png';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.heroBanner}>
      <div className={styles.container}>
        <Heading as="h1" className={styles.heroTitle} style={{ color: '#ffffff', fontWeight: 'bold' }}>
          {siteConfig.title}
        </Heading>
        <p className={styles.heroSubtitle}>
          {siteConfig.tagline}
        </p>
        <div className={styles.buttons} style={{ display: 'flex', gap: '1.2rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          <Link
            className="button button--primary button--lg"
            to="/docs/intro"
            style={{ fontWeight: 'bold', minWidth: '220px' }}>
            System Documentation
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="/docs/thesis"
            style={{ fontWeight: 'bold', minWidth: '220px' }}>
            Academic Thesis
          </Link>
        </div>
      </div>
    </header>
  );
}

function PipelineOverview() {
  return (
    <section className={styles.sectionAlt}>
      <div className={styles.container}>
        <div className={styles.sectionHeader}>
          <span style={{ color: 'var(--ifm-color-primary)', fontWeight: 'bold', fontSize: '0.85rem', letterSpacing: '2px', textTransform: 'uppercase' }}>Workflow Architecture</span>
          <Heading as="h2" className={styles.sectionTitle} style={{ marginTop: '0.5rem' }}>Detection Pipeline Overview</Heading>
          <p className={styles.sectionSubtitle}>Four integrated steps to analyze, filter, and classify threats in under 30 milliseconds.</p>
        </div>

        <div className={styles.pipelineFlow}>
          <div className={styles.pipelineStep}>
            <div className={styles.stepNumber}>01</div>
            <Heading as="h4" style={{ fontWeight: 'bold' }}>Visual Ingestion</Heading>
            <p style={{ fontSize: '0.9rem', color: 'var(--ifm-color-emphasis-700)', marginBottom: 0 }}>Captures live camera feeds or local security video streams at 30 FPS.</p>
          </div>
          <div className={styles.pipelineStep}>
            <div className={styles.stepNumber}>02</div>
            <Heading as="h4" style={{ fontWeight: 'bold' }}>Parallel YOLOv11</Heading>
            <p style={{ fontSize: '0.9rem', color: 'var(--ifm-color-emphasis-700)', marginBottom: 0 }}>Runs object detection for weapons and tracks 17-joint skeletal human poses.</p>
          </div>
          <div className={styles.pipelineStep}>
            <div className={styles.stepNumber}>03</div>
            <Heading as="h4" style={{ fontWeight: 'bold' }}>Torso Normalization</Heading>
            <p style={{ fontSize: '0.9rem', color: 'var(--ifm-color-emphasis-700)', marginBottom: 0 }}>Centers and scales skeleton coordinates relative to shoulders for distance invariance.</p>
          </div>
          <div className={styles.pipelineStep}>
            <div className={styles.stepNumber}>04</div>
            <Heading as="h4" style={{ fontWeight: 'bold' }}>LSTM Sequence HAR</Heading>
            <p style={{ fontSize: '0.9rem', color: 'var(--ifm-color-emphasis-700)', marginBottom: 0 }}>Classifies gestures over a 150-frame queue to flag active stabbing motions.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

function LiveDemoSection() {
  return (
    <section className={styles.section}>
      <div className={styles.container}>
        <div className={styles.grid2Col}>
          <div>
            <span style={{ color: 'var(--ifm-color-primary)', fontWeight: 'bold', fontSize: '0.85rem', letterSpacing: '2px', textTransform: 'uppercase' }}>Interactive GUI</span>
            <Heading as="h2" className={styles.sectionTitle} style={{ marginTop: '0.5rem' }}>Real-time Surveillance Monitor</Heading>
            <p style={{ fontSize: '1.05rem', lineHeight: '1.6', color: 'var(--ifm-color-emphasis-800)', marginBottom: '1.5rem' }}>
              The system features a lightweight Tkinter dashboard designed for operators. It overlays active tracking lines, weapon bounding boxes, and threat classifications (Safe vs. Danger) directly on the video feed.
            </p>
            <ul style={{ paddingLeft: '1.2rem', marginBottom: '2rem', color: 'var(--ifm-color-emphasis-700)', lineHeight: '1.6' }}>
              <li><strong>Dynamic Frame Skipping</strong>: Adjust processing load in real-time from 1 to 5 frames.</li>
              <li><strong>Automated Forensic Captures</strong>: Extracts and crops the face of suspects instantly.</li>
              <li><strong>Alert Logging</strong>: Appends detailed telemetry logs for security review.</li>
            </ul>
            <div style={{ display: 'flex', gap: '1rem' }}>
              <Link className="button button--primary" to="/docs/usage">Learn GUI Controls</Link>
              <a className="button button--secondary" href="https://www.youtube.com/playlist?list=PLo1f0U_Wr1t2QE8hyGbKqB8WCAeb5DOUH" target="_blank" rel="noopener noreferrer">
                Watch Demo Videos
              </a>
            </div>
          </div>

          <div className={styles.videoCard}>
            <img src={AppScreenshot} alt="Violence Detection GUI Screenshot" style={{ width: '100%', display: 'block' }} />
          </div>
        </div>
      </div>
    </section>
  );
}

function AcademicFoundation() {
  return (
    <section className={styles.sectionAlt}>
      <div className={styles.container}>
        <div className={styles.grid2Col}>
          <div className={styles.videoCard} style={{ background: '#0a0d14', padding: '2.5rem', border: '1px solid rgba(255,255,255,0.05)' }}>
            <span style={{ color: '#06b6d4', fontWeight: 'bold', fontSize: '0.75rem', letterSpacing: '1px', textTransform: 'uppercase' }}>Academic Abstract</span>
            <Heading as="h3" style={{ color: '#ffffff', marginTop: '0.5rem', marginBottom: '1.2rem', fontWeight: '700' }}>Research Foundation</Heading>
            <p style={{ color: '#94a3b8', fontSize: '0.95rem', lineHeight: '1.6', fontStyle: 'italic', marginBottom: '1.5rem' }}>
              "By separating visual features into parallel streams—object classification for weapons (YOLOv11 Knife) and skeletal keypoint estimation (YOLOv11-Pose)—the system isolates physical postures. These pose trajectories are normalized against perspective scaling and tracked across consecutive frames. The temporal sequences are then processed by a Bidirectional LSTM network."
            </p>
            <Link className="button button--outline button--secondary button--sm" to="/docs/thesis">Read Thesis Abstract</Link>
          </div>

          <div>
            <span style={{ color: 'var(--ifm-color-primary)', fontWeight: 'bold', fontSize: '0.85rem', letterSpacing: '2px', textTransform: 'uppercase' }}>Scientific Background</span>
            <Heading as="h2" className={styles.sectionTitle} style={{ marginTop: '0.5rem' }}>Thesis Paper Documentation</Heading>
            <p style={{ fontSize: '0.9rem', color: 'var(--ifm-color-primary)', fontWeight: 'bold', marginBottom: '1rem' }}>
              Author: Filippo Notari &bull; Advisor: Prof. Francesco Santini &bull; Università degli Studi di Perugia
            </p>
            <p style={{ fontSize: '1.05rem', lineHeight: '1.6', color: 'var(--ifm-color-emphasis-800)', marginBottom: '1.5rem' }}>
              The development of this real-time detection pipeline is backed by a structured academic thesis exploring human activity recognition (HAR), computer vision optimizations, and temporal modeling.
            </p>
            <p style={{ fontSize: '0.95rem', lineHeight: '1.6', color: 'var(--ifm-color-emphasis-700)', marginBottom: '2rem' }}>
              The documentation prepares all sections of the thesis, outlining research methodology, comparative models (CNN-LSTM vs. 3D-CNNs), training hyperparameters, and experimental accuracy outputs (AUC / Flicker Rates).
            </p>
            <Link className="button button--secondary" to="/docs/thesis">Explore Thesis & Download PDF</Link>
          </div>
        </div>
      </div>
    </section>
  );
}

function BottomCTA() {
  return (
    <section className={styles.bottomCta}>
      <div className={styles.container}>
        <Heading as="h2" style={{ fontSize: '2.25rem', fontWeight: 'bold', marginBottom: '1.5rem' }}>Deploy the Surveillance Pipeline</Heading>
        <p style={{ color: '#cbd5e1', fontSize: '1.1rem', maxWidth: '600px', margin: '0 auto 2.5rem auto', lineHeight: '1.5' }}>
          Explore the installation guides, prerequisites, and code directories to start tracking and classifying stabbing motions.
        </p>
        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
          <Link className="button button--primary button--lg" to="/docs/getting-started">Get Started</Link>
          <a className="button button--secondary button--lg" href="https://github.com/lraton/real-time-violent-action-detection" target="_blank" rel="noopener noreferrer">
            GitHub Repository
          </a>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title} - AI Video Surveillance`}
      description="Real-time surveillance system to detect violent stabbing actions and weapon presence using YOLOv11 and Bidirectional LSTM models.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <PipelineOverview />
        <LiveDemoSection />
        <AcademicFoundation />
        <BottomCTA />
      </main>
    </Layout>
  );
}
