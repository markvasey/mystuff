1. Epileptic Seizure Videos (Medical & Research)

For Tonic-Clonic (jerking) seizures, which are most relevant to your algorithm, these are
the best sources:

* VSViG Dataset: One of the few public research datasets that includes video. To protect
  privacy, the faces are blurred, but the body movements (skeletal data) are intact. You
  can find it on GitHub (VSViG) (https://github.com/vsvig-seizure/VSViG).
* ILAE (International League Against Epilepsy) Semiology: This is the gold standard for
  medical education. They have a Video Library
  (https://www.ilae.org/education/semiology-database) categorized by seizure type. Look
  for "Generalized Tonic-Clonic" or "Clonic" seizures to test your 2Hz–6Hz logic.
* Epilepsy Foundation - "What a Seizure Looks Like": They provide educational videos on
  YouTube specifically to help the public identify seizures.
    * Epilepsy Foundation YouTube Channel (https://www.youtube.com/@EpilepsyFoundation)
    * Search for: "Tonic-Clonic Seizure Demonstration" or "Grand Mal Seizure educational
      video."

2. Normal Activity Videos (Walking, Standing, Talking)

For "Negative" test cases (to ensure no false alarms), you should use standard computer
vision datasets. These are very easy to obtain:

* AVA (Atomic Visual Actions): This is the best dataset for "normal" life. It contains
  bounding boxes for people performing actions like standing, walking, sitting, and
  talking.
    * AVA Dataset Site (https://research.google.com/ava/)
* KTH Action Dataset: A classic, clean dataset that is perfect for initial validation. It
  contains 25 actors performing simple actions.
    * Walking, Standing, Hand Waving, Hand Clapping.
    * KTH Dataset Download (https://www.csc.kth.se/cvap/actions/)
* UCF101 / HMDB51: These are the industry standards for action recognition.
    * UCF101: Look for the "WalkingWithDog" or "Typing" categories.
    * HMDB51: Look for the "Walk", "Stand", or "Talk" categories.
