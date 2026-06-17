import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Visual Threat Analytics',
    tag: 'COMPUTER VISION',
    description: (
      <>
        Executes parallel YOLOv11 inference streams for real-time weapon (knife) classification and skeletal pose estimation, maintaining frame rates suitable for live video feeds.
      </>
    ),
  },
  {
    title: 'Temporal Sequence Processing',
    tag: 'DEEP LEARNING',
    description: (
      <>
        Integrates a Bidirectional LSTM network tracking a 150-frame temporal buffer of normalized keypoint velocities to classify suspicious physical gestures and stabbing movements.
      </>
    ),
  },
  {
    title: 'Automated Alert Logging',
    tag: 'SYSTEM INTEGRATION',
    description: (
      <>
        Triggers instant logging upon violent activity detection, extracting facial crops from clean video frames and archiving logs for forensic review.
      </>
    ),
  },
];

function Feature({tag, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center" style={{ marginBottom: '1.2rem' }}>
        <span style={{
          backgroundColor: 'var(--ifm-color-primary)',
          color: 'white',
          padding: '0.35rem 0.85rem',
          borderRadius: '4px',
          fontSize: '0.75rem',
          fontWeight: 'bold',
          letterSpacing: '1px'
        }}>
          {tag}
        </span>
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3" style={{ marginTop: '0.5rem', fontWeight: '600' }}>{title}</Heading>
        <p style={{ fontSize: '0.95rem', color: 'var(--ifm-color-emphasis-700)' }}>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features} style={{ padding: '4rem 0' }}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
