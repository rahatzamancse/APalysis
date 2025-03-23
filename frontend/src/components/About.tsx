import React from 'react'
import '@styles/about.css'

function About() {
  return (
    <div className="about-container">
      <h1 className="about-title">About The Project</h1>
      <p className="about-text">
        Paper Abstract goes here.
      </p>
      <h1>About Me</h1>
      <p>
        Rahat started his Ph.D. program at the University of Arizona with Dr. Stephen Kobourov (2021). Later, he transferred his Ph.D. program to the School of Computing at the University of Utah under supervision of Dr. Paul Rosen (2023).
      </p>
      <p>
        Rahat was born and raised in Bangladesh. He did his Bachelor of Science program with a thesis on “Deep Learning and Computer Vision” from Khulna University of Engineering & Technology (KUET) (2016-2020). After his BSc, he was an R&D intern at Ultra X Asia Pacific (2020). After that, he joined as a programmer at Bangladesh University of Engineering and Technology (BUET) (2021).
      </p>
      <p>
        <a href={"https://github.com/rahatzamancse"} target="_blank" rel="noopener noreferrer">
          My GitHub
        </a>
      </p>
      <p>
        <a href="https://rahatzamancse.netlify.app/" target="_blank" rel="noopener noreferrer">
          My Website
        </a>
      </p>
    </div>
  )
}

export default About